"""Generate the finalized leakage-safe PaySim v2 behavioural dataset."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

from src.features.behavioral import HISTORY_FEATURE_NAMES, calculate_bucket_history
from src.features.contract import REPOSITORY_ROOT, load_feature_contract


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest"]
DEFAULT_V1_ROOT = REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v1"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v2"
DEFAULT_WORK_ROOT = REPOSITORY_ROOT / "data/processed/.paysim_features_v2_work"


class V2BuildError(RuntimeError):
    """Raised when v2 cannot preserve identity, chronology, or reconciliation."""


def _raw_csv() -> Path:
    """Require one local raw PaySim file without embedding a machine path."""
    files = sorted((REPOSITORY_ROOT / "data/raw").glob("*.csv"))
    if len(files) != 1:
        raise V2BuildError(f"Expected one raw PaySim CSV, found {len(files)}")
    return files[0]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a bounded, index-free Parquet frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False),
        path,
        compression="snappy",
    )


def _bucket_raw_transactions(csv_path: Path, bucket_root: Path, buckets: int, chunk_size: int) -> int:
    """Partition by destination so exact history stays bounded to one disk bucket."""
    writers: dict[int, parquet.ParquetWriter] = {}
    source_row = 1
    try:
        for chunk_number, frame in enumerate(
            pd.read_csv(csv_path, usecols=RAW_COLUMNS, chunksize=chunk_size), start=1
        ):
            frame = frame.rename(
                columns={"type": "transaction_type", "nameOrig": "name_orig", "nameDest": "name_dest"}
            )
            frame.insert(0, "source_row_number", np.arange(source_row, source_row + len(frame)))
            source_row += len(frame)
            suffix = pd.to_numeric(frame["name_dest"].str.slice(1), errors="raise").astype("uint64")
            frame["_bucket"] = (suffix % buckets).astype("int16")
            for bucket, rows in frame.groupby("_bucket", sort=False):
                table = pa.Table.from_pandas(
                    rows.drop(columns="_bucket").reset_index(drop=True), preserve_index=False
                )
                bucket = int(bucket)
                if bucket not in writers:
                    path = bucket_root / f"bucket-{bucket:03d}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writers[bucket] = parquet.ParquetWriter(path, table.schema, compression="snappy")
                writers[bucket].write_table(table)
            LOGGER.info("Raw history bucket chunk %s written", chunk_number)
    finally:
        for writer in writers.values():
            writer.close()
    return source_row - 1


def _calculate_bucket_features(bucket_root: Path, fragment_root: Path, chunk_size: int) -> None:
    """Calculate destination groups independently and shard results by source chunk."""
    for bucket_path in sorted(bucket_root.glob("bucket-*.parquet")):
        result = calculate_bucket_history(pd.read_parquet(bucket_path))
        result["_part"] = ((result["source_row_number"] - 1) // chunk_size) + 1
        bucket_name = bucket_path.stem
        for part, rows in result.groupby("_part", sort=False):
            _write_frame(
                rows.drop(columns="_part"),
                fragment_root / f"part-{int(part):05d}" / f"{bucket_name}.parquet",
            )
        LOGGER.info("Behavioural features completed for %s", bucket_path.name)


def _v1_part_paths(v1_root: Path) -> list[tuple[str, Path]]:
    """Return v1 feature parts in deterministic chronological partition order."""
    paths: list[tuple[str, Path]] = []
    for partition in ("train", "validation", "test"):
        paths.extend((partition, path) for path in (v1_root / partition / "features").glob("part-*.parquet"))
    return sorted(paths, key=lambda item: (int(item[1].stem.split("-")[1]), ("train", "validation", "test").index(item[0])))


def _assemble_history_dataset(v1_root: Path, fragment_root: Path, output_root: Path) -> None:
    """Join numeric history to v1 rows while dropping raw grouping identifiers."""
    for partition, feature_path in _v1_part_paths(v1_root):
        name = feature_path.name
        metadata_path = v1_root / partition / "metadata" / name
        target_path = v1_root / partition / "target" / name
        metadata = pd.read_parquet(metadata_path)
        fragments = sorted((fragment_root / feature_path.stem).glob("*.parquet"))
        if not fragments:
            raise V2BuildError(f"Missing behavioural fragments for {name}")
        history = pd.concat((pd.read_parquet(path) for path in fragments), ignore_index=True)
        history = history.set_index("source_row_number")
        source_rows = metadata["source_row_number"].astype("int64")
        if not source_rows.isin(history.index).all():
            raise V2BuildError(f"Behavioural identity does not reconcile for {partition}/{name}")
        features = pd.concat(
            [
                pd.read_parquet(feature_path).reset_index(drop=True),
                history.loc[source_rows, list(HISTORY_FEATURE_NAMES)].reset_index(drop=True),
            ],
            axis=1,
        )
        _write_frame(features, output_root / partition / "features" / name)
        _write_frame(pd.read_parquet(target_path), output_root / partition / "target" / name)
        _write_frame(metadata, output_root / partition / "metadata" / name)


def build_v2_dataset(
    *,
    v1_root: Path = DEFAULT_V1_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    buckets: int = 64,
    chunk_size: int = 50_000,
) -> dict:
    """Generate all v2 partitions without overwriting reviewed local artifacts."""
    if output_root.exists() or work_root.exists():
        raise V2BuildError("v2 output or work directory already exists")
    if buckets <= 0 or chunk_size <= 0:
        raise ValueError("buckets and chunk_size must be positive")
    contract = load_feature_contract("config/features/paysim_v2.json")
    v1_manifest = json.loads((v1_root / "manifest.json").read_text(encoding="utf-8"))
    work_root.mkdir(parents=True)
    try:
        rows = _bucket_raw_transactions(_raw_csv(), work_root / "buckets", buckets, chunk_size)
        if rows != contract.expected_dataset.total_rows:
            raise V2BuildError(f"Raw row count {rows} does not reconcile")
        _calculate_bucket_features(work_root / "buckets", work_root / "fragments", chunk_size)
        _assemble_history_dataset(v1_root, work_root / "fragments", output_root)
        manifest = {
            **v1_manifest,
            "contract_version": contract.contract_version,
            "feature_schema": [
                {"name": feature.name, "dtype": feature.dtype} for feature in contract.features
            ],
            "history_policy": {
                "strictly_earlier_steps": True,
                "same_step_updates_deferred": True,
                "windows_hours": [24, 168],
                "raw_identifiers_predictive": False,
            },
            "excluded_candidates": {
                "relationship_features": "constant in PaySim train and validation",
                "isolation_forest": "not material at reviewed daily alert capacities",
            },
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise
    finally:
        if work_root.exists():
            # Windows/OneDrive can briefly retain Parquet handles after readers close.
            # Cleanup failure must be visible but must not invalidate a reconciled output.
            try:
                shutil.rmtree(work_root)
            except OSError as error:
                LOGGER.warning("Temporary v2 work directory needs manual cleanup: %s", error)


def main() -> int:
    """Run the full local v2 build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--log-level", choices=("INFO", "WARNING"), default="INFO")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        manifest = build_v2_dataset(buckets=arguments.buckets, chunk_size=arguments.chunk_size)
    except (OSError, ValueError, V2BuildError) as error:
        LOGGER.error("v2 dataset generation failed: %s", error)
        return 1
    LOGGER.info("v2 dataset completed: %s rows", manifest["expected_dataset"]["total_rows"] if "expected_dataset" in manifest else sum(v["rows"] for v in manifest["splits"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
