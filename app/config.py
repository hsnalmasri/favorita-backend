from pathlib import Path
import os

# WINDOWS path you gave. change later when deploying (or move to .env).
_DEFAULT_URL = "https://huggingface.co/datasets/hsnalmasri/favorita-demo/resolve/main/favorita_monthly_features.csv"

# read env var (optional), sanitize quotes/whitespace
_raw = (os.getenv("HF_DATASET_URL") or "").strip()
if (_raw.startswith('"') and _raw.endswith('"')) or (_raw.startswith("'") and _raw.endswith("'")):
    _raw = _raw[1:-1].strip()

# fall back to default if env empty
DATASET_PATH = _raw or _DEFAULT_URL  # <-- plain STRING (not Path)

# tell the app how to read your file (adjust to your columns)
DATE_COL   = "month"          # parsed as datetime
ITEM_COL   = "item_nbr"       # set to None to aggregate all items
TARGET_COL = "unit_sales"         # the value we forecast
OIL_DROP_COLS = ["dcoil_mean","dcoil_std","dcoil_last"]

# default ETS/Holt-Winters settings (safe starting point for monthly)
SEASONAL_PERIODS = 12
SEASONAL         = "add"     # "add" or "mul"
TREND            = "add"     # "add", "mul", or None

# API caps (guard rails)
MAX_HORIZON = 12
MIN_HISTORY_POINTS = 18      # require at least ~1.5 seasons of data
