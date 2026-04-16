import os
import torch


class Config:
    seed = 42

    path_to_train    = 'data/train.csv'
    path_to_test     = 'data/test.csv'
    path_to_checkpoints = 'checkpoints/'
    path_to_submission  = 'checkpoints/submission.csv'

    target_col = 'Survived'
    id_col     = 'PassengerId'

    n_splits = 5

    to_train = {
        'logreg':   True,
        'ridge':    True,
        'lasso':    True,
        'knn':      True,
        'dt':       True,
        'rf':       True,
        'catboost': True,
        'lgbm':     True,
        'xgb':      True,
        'dnn':      True,
    }

    logreg_params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': seed,
    }

    ridge_params = {
        'penalty': 'l2',
        'C': 1.0,
        'max_iter': 1000,
        'random_state': seed,
    }

    lasso_params = {
        'penalty': 'l1',
        'C': 0.1,
        'solver': 'saga',
        'max_iter': 2000,
        'random_state': seed,
    }

    knn_params = {
        'n_neighbors': 7,
        'weights': 'distance',
        'metric': 'euclidean',
    }

    dt_params = {
        'max_depth': 5,
        'min_samples_split': 5,
        'random_state': seed,
    }

    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': seed,
    }

    catboost_params = {
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'verbose': 0,
        'random_state': seed,
    }

    lgbm_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': seed,
    }

    xgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'random_state': seed,
        'eval_metric': 'logloss',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dnn_params = {
        'hidden_sizes':  [256, 128, 64],
        'dropout':       0.3,
        'activation':    'relu',           
        'epochs':        50,
        'batch_size':    32,
        'learning_rate': 1e-4,
        'weight_decay':  1e-4,             
        'patience':      15,               
        'scheduler':     'cosine',         
        'device':        device,
        'optimizer': 'AdamW'
    }

    meta_model_type   = 'logreg'     
    meta_model_params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': seed,
    }
    ensemble_mode = "all"