import os
import pickle
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from config import Config
from utils import print_scores, save_results_table, evaluate_ensembles, log_ensemble_results
from models.boosting import train_catboost, train_lgbm, train_xgb
from models.dnn import DNNClassifier
from sklearn.model_selection import cross_val_score


# Модели которым нужна нормализация через StandardScaler в solver
# (DNN делает нормализацию внутри себя)
NEEDS_SCALING = {'logreg', 'ridge', 'lasso', 'knn'}


class Solver:
    def __init__(self, config: Config) -> None:
        self.config  = config
        self.models  = {}
        self.scalers = {}
        self.results = {}
        self.meta_model = {}
        self._cached_ensemble_preds = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        os.makedirs(self.config.path_to_checkpoints, exist_ok=True)

        for model_name, should_train in self.config.to_train.items():
            if not should_train:
                continue
            print(f'\n{"="*50}')
            print(f'Обучаем: {model_name}  '
                  f'(device={self.config.device if model_name == "dnn" else "cpu"})')
            print('='*50)
            self._train_one(X, y, model_name)

        save_results_table(self.results, os.path.join(self.config.path_to_checkpoints, 'model_comparison.csv'), 
                            extra_params=self.config.dnn_params)
        return self.results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        all_probas = []

        for model_name, fold_models in self.models.items():
            fold_probas = []
            for fold_idx, model in enumerate(fold_models):
                X_input = X.copy()

                if model_name in NEEDS_SCALING:
                    scaler = self.scalers[model_name][fold_idx]
                    X_input = pd.DataFrame(
                        scaler.transform(X_input), columns=X.columns
                    )

                # DNN нормализует внутри predict_proba
                proba = model.predict_proba(X_input)[:, 1]
                fold_probas.append(proba)

            all_probas.append(np.mean(fold_probas, axis=0))

        avg_proba = np.mean(all_probas, axis=0)
        return (avg_proba >= 0.5).astype(int)
        
    def fit_stacking(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        assert len(self.results) > 0, 'Сначала вызови fit()'
        stacking_features = pd.DataFrame(index=X.index)
        if self.config.meta_model_type == 'logreg':
            lr_meta = LogisticRegression(**self.config.meta_model_params)
        else:
            raise ValueError(f'Неизвестный тип мета-модели: {self.config.meta_model_type}')
        for model_name, should_train in self.config.to_train.items():
            if not should_train:
                continue
            stacking_features[model_name] = self.results[model_name]['oof_preds']
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.seed,
        )
        scores = cross_val_score(lr_meta, stacking_features, y, cv=skf, scoring='accuracy')
        print("Stacking CV результат:")
        for i, score in enumerate(scores, 1):
            print(f"Fold {i}: {score:.4f}")
        print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        lr_meta.fit(stacking_features, y)
        self.meta_model = lr_meta
        return self.meta_model
    
    def predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        test_features = pd.DataFrame(index=X.index)
        for model_name, fold_models in self.models.items():
            fold_probas = []
            for fold_idx, model in enumerate(fold_models):
                X_input = X.copy()

                if model_name in NEEDS_SCALING:
                    scaler = self.scalers[model_name][fold_idx]
                    X_input = pd.DataFrame(
                        scaler.transform(X_input), columns=X.columns)
                proba = model.predict_proba(X_input)[:, 1]
                fold_probas.append(proba)
            mean_proba = np.mean(fold_probas, axis=0)
            test_features[model_name] = mean_proba
        final_pred = self.meta_model.predict(test_features)
        return final_pred
    
    def predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        all_probas = []
        norm_weights = {}
        acc_sum = sum(model['accuracy'] for model in self.results.values())
        for model_name, res in self.results.items():
            norm_weights[model_name] =  res['accuracy'] / acc_sum
        assert abs(sum(norm_weights.values()) - 1.0) < 1e-6, f'Сумма весов должна быть 1.0, получилось {sum(norm_weights.values()):.6f}'
        for model_name, fold_models in self.models.items():
            fold_probas = []
            for fold_idx, model in enumerate(fold_models):
                X_input = X.copy()

                if model_name in NEEDS_SCALING:
                    scaler = self.scalers[model_name][fold_idx]
                    X_input = pd.DataFrame(
                        scaler.transform(X_input), columns=X.columns
                    )
                proba = model.predict_proba(X_input)[:, 1]
                fold_probas.append(proba)
            avg_fold_proba = np.mean(fold_probas, axis=0)
            all_probas.append(avg_fold_proba * norm_weights[model_name])
        
        avg_proba = np.sum(all_probas, axis=0)
        return (avg_proba >= 0.5).astype(int)
    
    def fit_ensemble(self, X, y):
        if self.config.ensemble_mode in ["stacking", "all"]:
            self.fit_stacking(X, y)
        if self.config.ensemble_mode == "all":
            preds_dict = self.predict_ensemble(X)
            ensemble_scores = evaluate_ensembles(y, preds_dict)
            path = f"{self.config.path_to_checkpoints}ensemble_results.csv"
            log_ensemble_results(ensemble_scores, path)

    def predict_ensemble(self, X, use_cache: bool = True):
        if use_cache and self._cached_ensemble_preds is not None:
            return self._cached_ensemble_preds
        mode = self.config.ensemble_mode
        if mode == "voting":
            preds = self.predict(X)
        elif mode == "weighted":
            preds = self.predict_weighted(X)
        elif mode == "stacking":
            preds = self.predict_stacking(X)
        elif mode == "all":
            preds = {
                "voting": self.predict(X),
                "weighted": self.predict_weighted(X),
                "stacking": self.predict_stacking(X),
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if use_cache:
            self._cached_ensemble_preds = preds
        return preds
    
    def _build_model(self, model_name: str):
        cfg = self.config

        builders = {
            'logreg':   lambda: LogisticRegression(**cfg.logreg_params),
            'ridge':    lambda: LogisticRegression(**cfg.ridge_params),
            'lasso':    lambda: LogisticRegression(**cfg.lasso_params),
            'knn':      lambda: KNeighborsClassifier(**cfg.knn_params),
            'dt':       lambda: DecisionTreeClassifier(**cfg.dt_params),
            'rf':       lambda: RandomForestClassifier(**cfg.rf_params),
            'dnn':      lambda: DNNClassifier(**cfg.dnn_params),
        }
        return builders[model_name]() if model_name in builders else None

    def _train_fold(self, model_name: str, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple:
        scaler = None

        if model_name in NEEDS_SCALING:
            scaler = StandardScaler()
            X_tr  = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        if model_name == 'catboost':
            model = train_catboost(X_tr, y_tr, X_val, y_val, self.config.catboost_params)
        elif model_name == 'lgbm':
            model = train_lgbm(X_tr, y_tr, X_val, y_val, self.config.lgbm_params)
        elif model_name == 'xgb':
            model = train_xgb(X_tr, y_tr, X_val, y_val, self.config.xgb_params)
        elif model_name == 'dnn':
            # DNN получает X_val для early stopping
            model = self._build_model('dnn')
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        else:
            model = self._build_model(model_name)
            model.fit(X_tr, y_tr)

        return model, scaler

    def _train_one(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        if model_name == 'dnn':
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)

        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.seed,
        )

        fold_models, fold_scalers = [], []
        acc_list, auc_list, f1_list = [], [], []
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr  = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr  = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model, scaler = self._train_fold(model_name, X_tr, y_tr, X_val, y_val)

            X_val_input = X_val.copy()
            if scaler is not None:
                X_val_input = pd.DataFrame(
                    scaler.transform(X_val_input), columns=X.columns
                )

            preds = model.predict(X_val_input)
            proba = model.predict_proba(X_val_input)[:, 1]
            oof_preds[val_idx] = proba

            acc_list.append(accuracy_score(y_val, preds))
            auc_list.append(roc_auc_score(y_val, proba))
            f1_list.append(f1_score(y_val, preds))

            fold_models.append(model)
            fold_scalers.append(scaler)

            print(f'  Fold {fold+1}: acc={acc_list[-1]:.4f}  '
                  f'auc={auc_list[-1]:.4f}  f1={f1_list[-1]:.4f}')

        self.models[model_name]  = fold_models
        self.scalers[model_name] = fold_scalers
        self.results[model_name] = {
            'accuracy': np.mean(acc_list),
            'roc_auc':  np.mean(auc_list),
            'f1':       np.mean(f1_list),
            'acc_std':  np.std(acc_list),
            'oof_preds': oof_preds
        }
        scores_to_print = {k: v for k, v in self.results[model_name].items() if k != 'oof_preds'}
        print_scores(f'{model_name} ИТОГО', scores_to_print)

    def experiment_dnn(X, y):
        return None