import pytest

from src.data.postgres import load_postgres_settings


REQUIRED_VARIABLES = (
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
)


def test_load_postgres_settings(monkeypatch):
    values = {
        "POSTGRES_DB": "fraud_platform",
        "POSTGRES_USER": "fraud_user",
        "POSTGRES_PASSWORD": "local-test-password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = load_postgres_settings()

    assert settings.connection_kwargs() == {
        "dbname": "fraud_platform",
        "user": "fraud_user",
        "password": "local-test-password",
        "host": "localhost",
        "port": 5432,
    }
    assert "local-test-password" not in repr(settings)


def test_load_postgres_settings_lists_missing_values(monkeypatch):
    for name in REQUIRED_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="POSTGRES_DB.*POSTGRES_PASSWORD"):
        load_postgres_settings()


@pytest.mark.parametrize("port", ["not-a-number", "0", "65536"])
def test_load_postgres_settings_rejects_invalid_port(monkeypatch, port):
    for name in REQUIRED_VARIABLES:
        monkeypatch.setenv(name, "test")
    monkeypatch.setenv("POSTGRES_PORT", port)

    with pytest.raises(ValueError, match="POSTGRES_PORT"):
        load_postgres_settings()
