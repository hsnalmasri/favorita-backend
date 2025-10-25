# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import pandas as pd

from models.ets import ets_tune  # <- your backend function that returns (train_df, pred_df, metrics, res)

app = FastAPI(title="ETS API")

# allow Streamlit (localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- request / response schemas -----
Trend = Literal["add", "mul", None]
Seasonal = Literal["add", "mul", None]

class ETSTuneRequest(BaseModel):
    Ntest: int = Field(6, ge=1)
    horizon: int = Field(3, ge=1)
    CI: float = Field(0.95, gt=0, lt=1)
    trend: Trend = "add"
    seasonal: Seasonal = "mul"

class ETSTuneResponse(BaseModel):
    train: List[Dict[str, Any]]
    pred: List[Dict[str, Any]]
    metrics: Dict[str, float]

@app.post("/ets/tune", response_model=ETSTuneResponse)
def tune(req: ETSTuneRequest):
    # call your backend
    out = ets_tune(
        Ntest=req.Ntest,
        horizon=req.horizon,   # keep your existing arg name if it’s spelled this way in ets.py
        seasonal=req.seasonal,
        trend=req.trend,
        CI=req.CI,
    )

    train_df = out["train"]
    pred_df  = out["pred"]
    metrics  = {"smape": out["smape"], "mae": out["mae"], "bias": out["bias"]}
    res      = out["model"]   # keep server-side if you’ll generate future forecasts
    # make sure timestamps are JSON-serializable
    for df in (train_df, pred_df):
        if "ds" in df.columns:
            df = df.copy()
            df["ds"] = pd.to_datetime(df["ds"]).astype(str)

    return ETSTuneResponse(
        train=train_df.to_dict(orient="records"),
        pred=pred_df.to_dict(orient="records"),
        metrics={k: float(v) for k, v in metrics.items()},
    )
