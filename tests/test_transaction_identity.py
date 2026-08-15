import hashlib
import uuid

import pytest

from src.data.transaction_identity import calculate_dataset_sha256, create_transaction_id


def test_calculate_dataset_sha256_reads_in_chunks(tmp_path):
    dataset = tmp_path / "paysim.csv"
    content = b"step,type\n1,PAYMENT\n2,TRANSFER\n"
    dataset.write_bytes(content)

    assert calculate_dataset_sha256(dataset, chunk_size=3) == hashlib.sha256(content).hexdigest()


def test_create_transaction_id_is_deterministic_and_returns_uuid():
    dataset_id = "a" * 64

    first = create_transaction_id(dataset_id, 42)
    second = create_transaction_id(dataset_id, 42)

    assert first == second
    # This fixed example protects the cross-path identity contract from accidental changes.
    assert str(first) == "ad557565-983d-57b3-a07c-6a72b689ba68"
    assert isinstance(first, uuid.UUID)
    assert first.version == 5


def test_create_transaction_id_distinguishes_rows_with_identical_values():
    dataset_id = "b" * 64
    assert create_transaction_id(dataset_id, 10) != create_transaction_id(dataset_id, 11)


def test_create_transaction_id_distinguishes_dataset_versions():
    assert create_transaction_id("c" * 64, 10) != create_transaction_id("d" * 64, 10)


@pytest.mark.parametrize("dataset_id", ["", "A" * 64, "g" * 64, "a" * 63])
def test_create_transaction_id_rejects_invalid_dataset_fingerprint(dataset_id):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        create_transaction_id(dataset_id, 1)


@pytest.mark.parametrize("row_number", [0, -1])
def test_create_transaction_id_rejects_non_positive_rows(row_number):
    with pytest.raises(ValueError, match="greater than zero"):
        create_transaction_id("e" * 64, row_number)


@pytest.mark.parametrize("row_number", [True, 1.5, "1"])
def test_create_transaction_id_rejects_non_integer_rows(row_number):
    with pytest.raises(TypeError, match="must be an integer"):
        create_transaction_id("e" * 64, row_number)


def test_calculate_dataset_sha256_rejects_invalid_chunk_size(tmp_path):
    dataset = tmp_path / "paysim.csv"
    dataset.write_bytes(b"data")

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_dataset_sha256(dataset, chunk_size=0)
