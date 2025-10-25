import pandas as pd
from . import __init__ as _
from app.config import DATASET_PATH, DATE_COL, ITEM_COL, OIL_DROP_COLS


# choose columns only; give tight dtypes to cut RAM
_USECOLS = [DATE_COL, ITEM_COL, TARGET_COL]
_DTYPES  = {ITEM_COL: "int32", TARGET_COL: "float32"}  # or int32 if counts
_CHUNK   = 200_000  # tune if needed

def load_dataset(drop_last_month: bool = True) -> pd.DataFrame:
    print(f"[IO] Reading dataset from: {DATASET_PATH!r}")
    df = pd.read_csv(DATASET_PATH)

    if drop_last_month:
        dates=df[DATE_COL].unique()
        dates=sorted(dates, reverse=False)
        df=df[df[DATE_COL]<dates[-1]]
        
    
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.to_period("M").dt.to_timestamp("M")
      
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    df=df.drop(columns=OIL_DROP_COLS)
  
    return df

def load_monthly_total_agg(drop_last_month: bool = True) -> pd.DataFrame:
    """Stream the CSV in chunks and return tiny monthly total dataframe.
       Output: index=month (Timestamp, end of month), column='y' (float)
    """
    print(f"[IO] Streaming dataset from: {DATASET_PATH!r}")
    monthly_sum = None  # will hold a Series indexed by month

    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=_USECOLS,
        dtype=_DTYPES,
        chunksize=_CHUNK,
    ):
        # parse month lazily per chunk (avoid parse_dates overhead on all cols)
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        # group this chunk to monthly total (across ALL items)
        g = (
            chunk
            .groupby(chunk[DATE_COL].dt.to_period("M"))[TARGET_COL]
            .sum()
            .astype("float64")  # for statsmodels later
        )
        if monthly_sum is None:
            monthly_sum = g
        else:
            # align by index and add
            monthly_sum = monthly_sum.add(g, fill_value=0.0)

    if monthly_sum is None:
        raise ValueError("No data read from CSV (empty or wrong columns).")

    # convert PeriodIndex → Timestamp end-of-month; sort
    m = monthly_sum.sort_index()
    m.index = m.index.to_timestamp("M")

    if drop_last_month and len(m) > 0:
        m = m.iloc[:-1]

    # return as a tiny dataframe that ETS expects later
    out = pd.DataFrame({"ds": m.index, "y": m.values})
    out.set_index("ds", inplace=True)
    # enforce monthly freq to keep ETS happy
    out = out.asfreq("M")
    return out
