from __future__ import annotations

import io

import pandas as pd
import pytest

from dqcopilot.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the developer's real environment."""
    return Settings(
        app_env="ci",
        max_upload_mb=5,
        max_rows=10_000,
        max_columns=100,
        database_url="sqlite+pysqlite:///:memory:",
        anthropic_api_key=None,
    )


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A small frame containing one instance of each Milestone 1 problem."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005", "C002"],
            "name": ["Alice", " Bob", "bob", "Chloé", None, " Bob"],
            "country": ["FR", "fr", "FR", "  FR", "DE", "fr"],
            "notes": ["", "  ", None, "ok", None, "  "],
            "constant": ["X", "X", "X", "X", "X", "X"],
        }
    )


@pytest.fixture
def csv_bytes(messy_frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    messy_frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@pytest.fixture
def xlsx_bytes(messy_frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    messy_frame.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()
