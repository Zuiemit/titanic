import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model
