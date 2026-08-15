"""PostgreSQL connection configuration for local platform components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


_REQUIRED_ENVIRONMENT_VARIABLES = (
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
)


@dataclass(frozen=True)
class PostgresSettings:
    """Validated connection values loaded from environment configuration."""

    dbname: str
    user: str
    password: str = field(repr=False)
    host: str
    port: int

    def connection_kwargs(self) -> dict[str, str | int]:
        """Return keyword arguments accepted by Psycopg without logging secrets."""
        return {
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
        }


def load_postgres_settings() -> PostgresSettings:
    """Load required database settings and fail early with a useful message."""
    missing = [name for name in _REQUIRED_ENVIRONMENT_VARIABLES if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        port = int(os.environ["POSTGRES_PORT"])
    except ValueError as error:
        raise ValueError("POSTGRES_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("POSTGRES_PORT must be between 1 and 65535")

    return PostgresSettings(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=port,
    )
