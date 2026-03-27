from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path


def read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path
