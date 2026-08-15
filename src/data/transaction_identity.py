"""Create stable identities for PaySim rows across ingestion paths."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path


# Keeping this namespace fixed makes UUID generation reproducible in every environment.
PAYSIM_TRANSACTION_NAMESPACE = uuid.UUID("7d34554e-c520-5df0-a40c-4f7bc686010c")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def calculate_dataset_sha256(dataset_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the exact file fingerprint without loading the large PaySim CSV into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with Path(dataset_path).open("rb") as dataset_file:
        # Chunking keeps memory use constant for the 6.3-million-row source file.
        for chunk in iter(lambda: dataset_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_transaction_id(source_dataset_id: str, source_row_number: int) -> uuid.UUID:
    """Return the same logical ID for a PaySim file row in batch or Kafka ingestion."""
    if not _SHA256_PATTERN.fullmatch(source_dataset_id):
        raise ValueError("source_dataset_id must be a lowercase SHA-256 hexadecimal digest")
    if isinstance(source_row_number, bool) or not isinstance(source_row_number, int):
        raise TypeError("source_row_number must be an integer")
    if source_row_number <= 0:
        raise ValueError("source_row_number must be greater than zero")

    identity_name = f"sha256:{source_dataset_id}:row:{source_row_number}"
    return uuid.uuid5(PAYSIM_TRANSACTION_NAMESPACE, identity_name)
