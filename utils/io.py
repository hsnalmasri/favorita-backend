# app/utils/io.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

# If you use pyproject/packaging, keep this import so relative package init doesn't break
from . import __init__ as _  # noqa: F401

from app.config import (
    DATASET_PATH, DATE_COL, ITEM_COL, TARGET_COL, OIL_DROP_COLS
)

# Optional: lightweight memory logging (safe no-op if psutil is missing)
def _mem_log(tag: str):
    try:
        import os, psutil, gc
        gc.collect()
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        print(f"[MEM] {tag:20s} => {rss:8.1f} MB")
    except Exception:
        pass


# ---------- cache paths ----------
_CACHE_DIR = Path(os.environ.get("APP_CACHE_DIR", "/tmp/favorita_cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _parquet_cache_path() -> Path:
    # name cache file after the source, but as .parquet
    src = str(DATASET_PATH)
    base = Path(src).name  # works for URLs too
    if base.endswith(".csv"):
        base = base[:-4]
    return _CACHE_DIR / f"{base}.parquet"


# ---------- CSV -> Parquet (chunked, low-RAM) ----------
def _csv_to_parquet_chunked(
    csv_path_or_url: str | Path,
    parquet_path: Path,
    chunksize: int = 200_000,
):
    """
    Create Parquet from CSV without loading the entire file in memory.
    Only keeps the minimal columns we actually use downstream.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    usecols = [c for c in [DATE_COL, ITEM_COL, TARGET_COL] if c is not None]
    dtypes = {}
    if ITEM_COL is not None:
        # categories keep memory low; will become string in Parquet
        dtypes[ITEM_COL] = "category"

    writer = None
    part_idx = 0

    for chunk in pd.read_csv(
        csv_path_or_url,
        usecols=usecols,
        dtype=dtypes or None,
        chunksize=chunksize,
    ):
        part_idx += 1
        # Normalize month to month-end timestamps (matches your current pipeline)
        if not pd.api.types.is_datetime64_any_dtype(chunk[DATE_COL]):
            chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce") \
                                 .dt.to_period("M").dt.to_timestamp("M")

        # Drop any rows with missing month or target
        chunk = chunk.dropna(subset=[DATE_COL, TARGET_COL])

        # Convert to Arrow table and stream-write to Parquet (no big concat)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(parquet_path.as_posix(), table.schema)
        writer.write_table(table)

        _mem_log(f"chunk {part_idx} written")

    if writer is not None:
        writer.close()


# ---------- public loaders ----------
def load_dataset(
    drop_last_month: bool = True,
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    """
    Load the base dataset with minimal RAM.
    Priority:
      1) Cached Parquet at /tmp/favorita_cache/*.parquet
      2) If not found and source is CSV -> build Parquet chunked, then read
      3) If source itself is Parquet -> read directly
    Post-processing mirrors your old loader:
      - Ensure month is month-end timestamp
      - Cast item to string
      - Drop OIL_DROP_COLS if present
      - Optionally drop the last month
    """
    src = str(DATASET_PATH)
    parquet_path = _parquet_cache_path()

    df: pd.DataFrame | None = None

    if prefer_parquet and parquet_path.exists():
        _mem_log("read_parquet(cache) start")
        # Read only useful columns if Parquet schema has extras
        columns = [c for c in [DATE_COL, ITEM_COL, TARGET_COL] if c is not None]
        df = pd.read_parquet(parquet_path, columns=columns)
        _mem_log("read_parquet(cache) done")

    elif src.lower().endswith(".parquet"):
        _mem_log("read_parquet(src) start")
        columns = [c for c in [DATE_COL, ITEM_COL, TARGET_COL] if c is not None]
        # pandas can read remote parquet if fsspec is available; otherwise place file locally.
        df = pd.read_parquet(src, columns=columns)
        _mem_log("read_parquet(src) done")

    else:
        # Source is CSV (possibly a URL). Build a local Parquet cache in a streaming way.
        _mem_log("build_parquet from CSV start")
        _csv_to_parquet_chunked(src, parquet_path)
        _mem_log("build_parquet from CSV done")
        _mem_log("read_parquet(cache) start")
        columns = [c for c in [DATE_COL, ITEM_COL, TARGET_COL] if c is not None]
        df = pd.read_parquet(parquet_path, columns=columns)
        _mem_log("read_parquet(cache) done")

    # ----- Post-processing (mirror your previous logic) -----
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce") \
                           .dt.to_period("M").dt.to_timestamp("M")

    if ITEM_COL is not None:
        # Ensure consistent string type for item ids
        df[ITEM_COL] = df[ITEM_COL].astype(str)

    # Drop oil columns if they happen to exist in Parquet (safe if missing)
    for c in OIL_DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # Optionally drop the last (most recent) month (matches your old behavior)
    if drop_last_month:
        dates = sorted(df[DATE_COL].dropna().unique())
        if dates:
            df = df[df[DATE_COL] < dates[-1]]

    return df


def load_monthly_total(drop_last_month: bool = True) -> pd.DataFrame:
    """
    Fast path for monthly total sales:
      - Ensures Parquet cache exists (created chunked from CSV if needed)
      - Reads only DATE_COL & TARGET_COL from Parquet
      - Aggregates by month and returns an index = month DatetimeIndex
    This avoids loading item-level detail in memory.
    """
    src = str(DATASET_PATH)
    parquet_path = _parquet_cache_path()

    if not parquet_path.exists():
        _mem_log("monthly_total: build_parquet start")
        _csv_to_parquet_chunked(src, parquet_path)
        _mem_log("monthly_total: build_parquet done")

    cols = [DATE_COL, TARGET_COL]
    _mem_log("monthly_total: read_parquet start")
    df = pd.read_parquet(parquet_path, columns=cols)
    _mem_log("monthly_total: read_parquet done")

    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce") \
                           .dt.to_period("M").dt.to_timestamp("M")

    if drop_last_month:
        dates = sorted(df[DATE_COL].dropna().unique())
        if dates:
            df = df[df[DATE_COL] < dates[-1]]

    g = (
        df.groupby([DATE_COL])[TARGET_COL]
          .sum()
          .reset_index()
          .rename(columns={TARGET_COL: "unit_sales"})
    )

    g.set_index(DATE_COL, inplace=True)
    return g
