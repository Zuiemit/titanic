import os
import random
import numpy as np
import pandas as pd
import torch
import csv
from sklearn.metrics import accuracy_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def print_scores(model_name: str, scores: dict) -> None:
    print(f'\n{model_name}')
    print(f'  accuracy = {scores["accuracy"]:.4f} ± {scores["acc_std"]:.4f}')
    print(f'  roc_auc  = {scores["roc_auc"]:.4f}')
    print(f'  f1       = {scores["f1"]:.4f}')

def save_results_table(all_results: dict, path: str, extra_params: dict = None) -> None:
    rows = []
    for name, res in all_results.items():
        rows.append({
            'model':    name,
            'accuracy': round(res['accuracy'], 4),
            'roc_auc':  round(res['roc_auc'], 4),
            'f1':       round(res['f1'], 4)
        })
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    df.to_csv(path, index=False)
    print(f'\nТаблица результатов сохранена: {path}')
    print(df.to_string(index=False))

    if extra_params is not None:
        log_path = path.replace('.csv', 'res_experiments.csv')
        dnn_scores = all_results.get('dnn', {})
        row = {
            **extra_params,
            'accuracy': round(dnn_scores.get('accuracy', 0), 4),
            'roc_auc':  round(dnn_scores.get('roc_auc',  0), 4),
            'f1':       round(dnn_scores.get('f1',       0), 4),
        }
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f'Лог эксперимента дописан: {log_path}')

def log_ensemble_results(results: dict, path: str) -> None:
    rows = []
    for name, score in results.items():
        rows.append({
            'ensemble': name,
            'accuracy': round(score['accuracy'], 4),
        })
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    df.to_csv(path, index=False)
    print(f'\nРезультаты ансамблей сохранены: {path}')
    print(df.to_string(index=False))

def evaluate_ensembles(y_true, preds_dict: dict) -> dict:
    results = {}
    print("\n=== Ensemble comparison ===")
    for name, preds in preds_dict.items():
        acc = accuracy_score(y_true, preds)
        results[name] = {'accuracy': acc}
        print(f"{name}: {acc:.4f}")
    return results