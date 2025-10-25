import pandas as pd
import numpy as np

def smape(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, (np.abs(y_true)+np.abs(y_pred)))
    score=100*np.mean(2.0*np.abs(y_pred-y_true)/denom)
    return score

def mean_bias_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.nanmean(y_pred - y_true)  # >0 over-forecast, <0 under-forecast