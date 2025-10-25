import pandas as pd
from . import __init__ as _
from app.config import DATASET_PATH, DATE_COL, ITEM_COL, OIL_DROP_COLS

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
