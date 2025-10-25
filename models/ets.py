import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from utils import io
from app.config import DATE_COL, ITEM_COL, TARGET_COL, SEASONAL_PERIODS, SEASONAL, TREND, MIN_HISTORY_POINTS
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import err_metrics

class NotEnoughHistory(Exception): ...

def _to_series(df: pd.DataFrame) -> pd.Series:
    s=pd.Series(df[TARGET_COL].values, index=pd.DatetimeIndex(df.index.values))
    print("transforming to series")
    s =s.asfreq('ME')
    return s

def _pick_slice(df: pd.DataFrame, item_id: str | None) -> pd.DataFrame:
    print('picking a slice')
    if item_id is not None:
        df = df[df[ITEM_COL] == item_id]
        print("picked a slice:", item_id)
    elif item_id is None:
        print("no slice picked")
        pass  # already aggregated scenario
    print('finished picking a slice')

    return df

def _aggregate_df(df: pd.DataFrame, agg_all: bool = True) -> pd.DataFrame:
    print('aggregating if needed')
    print("checking dates compatibility")

    

    print("dates checking complete")

    if agg_all:
        print("aggregating all")
        g = df.groupby([DATE_COL])['unit_sales'].sum().reset_index()
        g[DATE_COL] = pd.to_datetime(g[DATE_COL], errors='coerce')
        g.set_index(DATE_COL, inplace=True)
        print("all aggregated, returning")
        return g
    print('Aggregating to ITEM leveles')
    g=df.groupby([DATE_COL, ITEM_COL])['unit_sales'].sum().reset_index()
    g[DATE_COL] = pd.to_datetime(g[DATE_COL], errors='coerce')
    g.set_index(DATE_COL, inplace=True)
    print("ITEM aggregated, returning")
    return g

def train_test_dataset(df:pd.DataFrame, Ntest:int=6)->pd.DataFrame:
    dates=sorted(df.index.unique(), reverse=False)
    train_dates=dates[:-Ntest]
    test_dates=dates[-Ntest:]
    train_set=df[df.index.isin(train_dates)].copy()
    test_set=df[df.index.isin(test_dates)].copy()
    return train_set, test_set


def monthly_total_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    print("importing dataset")
    df = io.load_dataset()
    print("aggregating monthly total sales")
    df = df.groupby([DATE_COL])['unit_sales'].sum().reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.to_period("M").dt.to_timestamp("M")
    df.set_index(DATE_COL, inplace=True)
    print("aggregating successfull")
    return df

def ets_tune(Ntest: int = 6,
    horizon: int = 6,
    # model spec
    error: str = "add",                 # "add" | "mul"
    trend: str | None = TREND,          # "add" | "mul" | None
    seasonal: str | None = SEASONAL,    # "add" | "mul" | None
    seasonal_periods: int = SEASONAL_PERIODS,
    damped_trend: bool = False,
    # manual coefficients (None => auto)
    smoothing_level: float | None = None,      # alpha
    smoothing_trend: float | None = None,      # beta
    smoothing_seasonal: float | None = None,   # gamma
    phi: float | None = None,                  # damping_trend coefficient φ
    # eval
    item_id: str | None = None,
    CI: float = 0.95,
)->pd.DataFrame:
    
    df=io.load_dataset()
    df=monthly_total_aggregate(df)
    train_set, test_set=train_test_dataset(df, Ntest)
    
    train_set=_to_series(train_set)
    test_set=_to_series(test_set)
    
    print('fitting ETS model')
    model = ETSModel(
        train_set,
        error=error,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )

    all_manual = (
        smoothing_level is not None
        and (trend is None or smoothing_trend is not None)        # beta only matters if trend
        and (seasonal is None or smoothing_seasonal is not None)  # gamma only if seasonal
        and (not damped_trend or phi is not None)                 # phi only if damped
    )

    res = model.fit(
        optimized=not all_manual,
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        damping_trend=phi,   # φ
    )

    fitted=res.fittedvalues
    start_date=test_set.index[-1]+test_set.index.freq
    end_date=test_set.index[-1]+test_set.index.freq*Ntest

    print("forecasting")
    pred=res.get_prediction(start=start_date, end=end_date, optimized=True)

    print("forecast successfull")
    sf=pred.summary_frame(alpha=1-CI)
    mean=sf.filter(like="mean").iloc[:, 0]
    lower=sf.filter(like="lower").iloc[:, 0]
    upper=sf.filter(like="upper").iloc[:, 0]

    print('evaluating ETS model')
    smape=err_metrics.smape(test_set.values, mean)
    mae=mean_absolute_error(test_set.values, mean)
    bias=err_metrics.mean_bias_error(test_set.values, mean)
    

    train_out=pd.DataFrame({
        'ds':train_set.index.values,
        'y':train_set.values,
        'fitted':fitted.values
    })
    pred_out=pd.DataFrame({
        'ds':test_set.index.values,
        'y':test_set.values,
        'yhat':mean,
        'lower':lower,
        'upper':upper
    })
    return {
        'train':train_out,
        'pred':pred_out,
        'smape':smape,
        'mae':mae,
        'bias':bias,
        'model': res
        
    }
    

def fit_and_forecast(
        df: pd.DataFrame,
        horizon: int, 
        item_id: str | None = None,
        seasonal: str = SEASONAL,
        trend: str = TREND,
        seasonal_periods: int = SEASONAL_PERIODS
):
    print("ETS started")
    df = _aggregate_df(df)
    df = _pick_slice(df, item_id)

    if len(df) < MIN_HISTORY_POINTS:
        raise NotEnoughHistory(f'need at least {MIN_HISTORY_POINTS} points, got {len(df)}')
    
    y = _to_series(df)
    y = y.asfreq("M").astype("float64")

    print("fitting ETS model")

    model = ETSModel(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )

    res = model.fit(optimized=True)

    print("predicting values")
    start=y.index[-1]+y.index.freq
    end=y.index[-1]+y.index.freq*horizon
    pred = res.get_prediction(start=start, end=end, optimized=True)
    print("predictiong finished")

    sf = pred.summary_frame(alpha=0.05)

    mean = sf.filter(like="mean").iloc[:, 0]
    lower = sf.filter(like="lower").iloc[:, 0]
    upper = sf.filter(like="upper").iloc[:, 0]

    
    resid = y - res.fittedvalues.reindex(y.index)

    sigma = float(np.nanstd(resid, ddof=1)) if resid.notna().any() else 0.0

    out = pd.DataFrame({
        'ds' : mean.index,
        'yhat' : mean.values,
        'yhat_lower' : lower.values,
        'yhat_upper' : upper.values,
    }).astype({'ds':'datetime64[ns]'})

    fitted = pd.DataFrame({
        'ds' : res.fittedvalues.index.astype('datetime64[ns]'),
        'y_fitted' : res.fittedvalues.values,
    })

    return {
        
        "fitted": fitted,
        "forecast": out,
        "params": {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "sigma": sigma,
            "aic": getattr(res, "aic", None),
            "bic": getattr(res, "bic", None),
            "model" : res
        },
    }

def ets_from_model(model: ETSModel, df: pd.DataFrame, horizon: int, CI: float = 0.95):
    print("ETS started")
    print("aggregating dataset")
    df = _aggregate_df(df)
    y = _to_series(df)

    print("predicting values")
    start=y.index[-1]+y.index.freq
    end=y.index[-1]+y.index.freq*horizon
    pred = model.get_prediction(start=start, end=end, optimized=True)
    print("predictiong finished")

    sf = pred.summary_frame(alpha=1-CI)

    mean = sf.filter(like="mean").iloc[:, 0]
    lower = sf.filter(like="lower").iloc[:, 0]
    upper = sf.filter(like="upper").iloc[:, 0]

    out = pd.DataFrame({
        'ds' : mean.index,
        'yhat' : mean.values,
        'yhat_lower' : lower.values,
        'yhat_upper' : upper.values,
    }).astype({'ds':'datetime64[ns]'})

    return out