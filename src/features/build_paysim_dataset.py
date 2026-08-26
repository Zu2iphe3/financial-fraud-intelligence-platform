"""Generate versioned, chronological PaySim modelling datasets from PostgreSQL."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.parquet as parquet
from psycopg import sql
from psycopg.rows import dict_row

from src.data.postgres import load_postgres_settings
from src.features.contract import (
    DEFAULT_CONTRACT_PATH,
    REPOSITORY_ROOT,
    FeatureContract,
    load_feature_contract,
)
from src.features.pipeline import ModellingChunk, prepare_modelling_chunk


LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "processed" / "paysim_features"


class DatasetBuildError(RuntimeError):
    """Raised when generated partitions do not reconcile with their source contract."""


def _source_query(contract: FeatureContract) -> sql.Composed:
    """Select only contracted columns in deterministic source-row order."""
    schema_name, relation_name = contract.source_relation.split(".", maxsplit=1)
    columns = sql.SQL(", ").join(map(sql.Identifier, contract.required_source_columns))
    return sql.SQL(
        "SELECT {} FROM {}.{} ORDER BY step, source_dataset_id, source_row_number"
    ).format(columns, sql.Identifier(schema_name), sql.Identifier(relation_name))


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write one bounded frame as compressed, cross-platform Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    parquet.write_table(table, path, compression="snappy")


def _write_chunk(
    output_directory: Path,
    chunk_number: int,
    modelling_chunk: ModellingChunk,
) -> None:
    """Use matching part names so features, targets, and metadata stay aligned."""
    for partition in ("train", "validation", "test"):
        mask = modelling_chunk.metadata["partition"].eq(partition)
        if not mask.any():
            continue
        part_name = f"part-{chunk_number:05d}.parquet"
        _write_frame(
            modelling_chunk.features.loc[mask],
            output_directory / partition / "features" / part_name,
        )
        _write_frame(
            modelling_chunk.target.loc[mask].to_frame(),
            output_directory / partition / "target" / part_name,
        )
        _write_frame(
            modelling_chunk.metadata.loc[mask],
            output_directory / partition / "metadata" / part_name,
        )


def _empty_counts(contract: FeatureContract) -> dict[str, dict[str, int | None]]:
    """Create explicit counters used to reconcile every chronological split."""
    return {
        split.name: {
            "rows": 0,
            "fraud_rows": 0,
            "first_step": None,
            "last_step": None,
            "final_day_rows": 0,
            "final_day_fraud_rows": 0,
        }
        for split in contract.splits
    }


def _update_counts(
    counts: dict[str, dict[str, int | None]],
    modelling_chunk: ModellingChunk,
    contract: FeatureContract,
) -> None:
    """Accumulate bounded-chunk facts without retaining earlier rows in memory."""
    for split in contract.splits:
        mask = modelling_chunk.metadata["partition"].eq(split.name)
        if not mask.any():
            continue
        partition_metadata = modelling_chunk.metadata.loc[mask]
        partition_target = modelling_chunk.target.loc[mask]
        current = counts[split.name]
        current["rows"] = int(current["rows"] or 0) + int(mask.sum())
        current["fraud_rows"] = int(current["fraud_rows"] or 0) + int(
            partition_target.sum()
        )
        observed_first = int(partition_metadata["step"].min())
        observed_last = int(partition_metadata["step"].max())
        current["first_step"] = (
            observed_first
            if current["first_step"] is None
            else min(int(current["first_step"]), observed_first)
        )
        current["last_step"] = (
            observed_last
            if current["last_step"] is None
            else max(int(current["last_step"]), observed_last)
        )
        final_mask = partition_metadata["evaluation_cohort"].eq(
            contract.final_day.evaluation_cohort
        )
        current["final_day_rows"] = int(current["final_day_rows"] or 0) + int(
            final_mask.sum()
        )
        current["final_day_fraud_rows"] = int(
            current["final_day_fraud_rows"] or 0
        ) + int(partition_target.loc[final_mask].sum())


def _validate_summary(
    contract: FeatureContract,
    split_counts: dict[str, dict[str, int | None]],
    source_dataset_ids: set[str],
    validate_expected_facts: bool,
) -> None:
    """Reject partial, mixed-source, or factually inconsistent dataset builds."""
    problems: list[str] = []
    if len(source_dataset_ids) != 1:
        problems.append(
            "source must contain exactly one dataset fingerprint; found "
            f"{len(source_dataset_ids)}"
        )
    for split in contract.splits:
        observed = split_counts[split.name]
        if observed["first_step"] != split.first_step or observed["last_step"] != split.last_step:
            problems.append(
                f"{split.name} observed steps {observed['first_step']}..{observed['last_step']} "
                f"do not match {split.first_step}..{split.last_step}"
            )

    if validate_expected_facts:
        expected = contract.expected_dataset
        total_rows = sum(int(values["rows"] or 0) for values in split_counts.values())
        fraud_rows = sum(
            int(values["fraud_rows"] or 0) for values in split_counts.values()
        )
        test_counts = split_counts["test"]
        observed_facts = (
            total_rows,
            fraud_rows,
            int(test_counts["final_day_rows"] or 0),
            int(test_counts["final_day_fraud_rows"] or 0),
        )
        expected_facts = (
            expected.total_rows,
            expected.fraud_rows,
            expected.final_day_rows,
            expected.final_day_fraud_rows,
        )
        if observed_facts != expected_facts:
            problems.append(
                "dataset facts do not reconcile: "
                f"observed={observed_facts}, expected={expected_facts}"
            )
        for split in contract.splits:
            observed = split_counts[split.name]
            observed_partition = (
                int(observed["rows"] or 0),
                int(observed["fraud_rows"] or 0),
            )
            expected_partition = (split.expected_rows, split.expected_fraud_rows)
            if observed_partition != expected_partition:
                problems.append(
                    f"{split.name} counts do not reconcile: "
                    f"observed={observed_partition}, expected={expected_partition}"
                )
    if problems:
        raise DatasetBuildError("Dataset validation failed: " + "; ".join(problems))


def _manifest(
    contract: FeatureContract,
    split_counts: dict[str, dict[str, int | None]],
    source_dataset_ids: set[str],
) -> dict[str, Any]:
    """Record the exact schema, source identity, partitions, and artifact policy."""
    return {
        "contract_version": contract.contract_version,
        "source_relation": contract.source_relation,
        "source_dataset_id": next(iter(source_dataset_ids)),
        "feature_schema": [
            {"name": feature.name, "dtype": feature.dtype}
            for feature in contract.features
        ],
        "target": contract.target_column,
        "metadata_columns": [
            *contract.metadata_columns,
            "partition",
            "evaluation_cohort",
        ],
        "splits": split_counts,
        "final_day_policy": {
            "simulation_day": contract.final_day.simulation_day,
            "first_step": contract.final_day.first_step,
            "last_step": contract.final_day.last_step,
            "evaluation_cohort": contract.final_day.evaluation_cohort,
            "predictive_feature": False,
        },
    }


def build_modelling_dataset(
    connection: psycopg.Connection,
    contract: FeatureContract,
    output_directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    validate_expected_facts: bool = True,
) -> dict[str, Any]:
    """Stream the trusted mart into separated, versioned Parquet partitions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise DatasetBuildError(
            f"Output directory already exists: {destination}. Remove it before rebuilding."
        )
    destination.mkdir(parents=True)

    split_counts = _empty_counts(contract)
    source_dataset_ids: set[str] = set()
    try:
        # A server-side cursor keeps memory bounded while preserving deterministic source order.
        with connection.cursor(
            name="paysim_feature_source", row_factory=dict_row
        ) as cursor:
            cursor.execute(_source_query(contract))
            chunk_number = 0
            while rows := cursor.fetchmany(chunk_size):
                chunk_number += 1
                source = pd.DataFrame(rows)
                modelling_chunk = prepare_modelling_chunk(source, contract)
                source_dataset_ids.update(
                    str(value) for value in source["source_dataset_id"].unique()
                )
                _write_chunk(destination, chunk_number, modelling_chunk)
                _update_counts(split_counts, modelling_chunk, contract)
                LOGGER.info("Feature chunk %s written: rows=%s", chunk_number, len(source))

        _validate_summary(
            contract, split_counts, source_dataset_ids, validate_expected_facts
        )
        manifest = _manifest(contract, split_counts, source_dataset_ids)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        # Only the directory created by this invocation is removed; an existing output is
        # never overwritten, protecting previously reviewed modelling data.
        shutil.rmtree(destination)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the repository-relative command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    """Generate the configured modelling dataset and return a shell-friendly status."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        contract = load_feature_contract(arguments.contract)
        output_directory = Path(arguments.output_root) / contract.contract_version
        settings = load_postgres_settings()
        with psycopg.connect(**settings.connection_kwargs()) as connection:
            manifest = build_modelling_dataset(
                connection,
                contract,
                output_directory,
                arguments.chunk_size,
            )
    except (DatasetBuildError, ValueError, psycopg.Error) as error:
        LOGGER.error("Modelling dataset generation failed: %s", error)
        return 1

    total_rows = sum(int(values["rows"]) for values in manifest["splits"].values())
    LOGGER.info(
        "Modelling dataset completed: contract=%s, rows=%s, output=%s",
        contract.contract_version,
        f"{total_rows:,}",
        output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
