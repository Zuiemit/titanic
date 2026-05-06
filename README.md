# Titanic

---

## Структура проекта

```
titanic/
├── data/
│   ├── train.csv
│   └── test.csv
├── checkpoints/
│   ├── submission.csv          # финальные предсказания
│   ├── model_comparison.csv    # метрики всех моделей
│   └── ensemble_results.csv    # сравнение методов ансамблирования
├── models/
│   ├── __init__.py
│   ├── boosting.py             # CatBoost, LightGBM, XGBoost
│   └── dnn.py                  # PyTorch MLP
├── config.py                   # все настройки проекта
├── dataset.py                  # предобработка данных
├── solver.py                   # обучение, валидация, ансамбли
├── utils.py                    # вспомогательные функции
├── main.py                     # точка входа
└── requirements.txt
```

---

## Быстрый старт

**1. Установить зависимости**

```bash
pip install -r requirements.txt
```

**2. Данные**

Данные `train.csv` и `test.csv` находятся в папке `data/`.

**3. Запустить**

```bash
python main.py
```

После запуска в `checkpoints/` появятся:
- `submission.csv` — файл для загрузки на Kaggle
- `model_comparison.csv` — таблица метрик всех моделей
- `ensemble_results.csv` — сравнение методов ансамблирования

---

## Модели

Все модели включаются/выключаются в `config.py` через словарь `to_train`:

```python
to_train = {
    'logreg':   True,   # Logistic Regression
    'ridge':    False,  # Ridge (L2)
    'lasso':    False,  # Lasso (L1)
    'knn':      False,  # K-Nearest Neighbors
    'dt':       False,  # Decision Tree
    'rf':       False,  # Random Forest
    'catboost': True,   # CatBoost
    'lgbm':     False,  # LightGBM
    'xgb':      False,  # XGBoost
    'dnn':      True,   # Deep Neural Network (PyTorch)
}
```

---

## Ансамбли

Режим ансамблирования задаётся в `config.py`:

```python
ensemble_mode = "all"   # "voting" | "weighted" | "stacking" | "all"
```

| Режим | Описание |
|---|---|
| `voting` | Soft voting |
| `weighted` | Взвешенное усреднение |
| `stacking` | Стекинг — мета-модель (LogReg) |
| `all` | Запускает все три и сравнивает результаты |

---

## Настройка параметров

Все параметры моделей задаются в `config.py`. Пример для CatBoost:

```python
catboost_params = {
    'iterations':    300,
    'learning_rate': 0.05,
    'depth':         6,
    'verbose':       0,
    'random_state':  seed,
}
```

Параметры DNN:

```python
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
    'optimizer':     'AdamW',         
    'device':        'cpu',           
}
```

---

## Логирование экспериментов

При каждом запуске результаты DNN автоматически дописываются в `checkpoints/model_comparisonres_experiments.csv`. Это позволяет сравнивать эксперименты с разными параметрами.

---

## Результаты

==================================================
Обучаем: xgb  (device=cpu)
==================================================
  Fold 1: acc=0.8715  auc=0.9132  f1=0.8217
  Fold 2: acc=0.8596  auc=0.8941  f1=0.8092
  Fold 3: acc=0.8034  auc=0.8458  f1=0.7200
  Fold 4: acc=0.8315  auc=0.8807  f1=0.7692
  Fold 5: acc=0.8371  auc=0.8876  f1=0.7752

xgb ИТОГО
  accuracy = 0.8406 ± 0.0236
  roc_auc  = 0.8843
  f1       = 0.7791

...

Таблица результатов сохранена: checkpoints/model_comparison.csv
   model  accuracy  roc_auc     f1
    lgbm    0.8406   0.8843 0.7828
     xgb    0.8406   0.8843 0.7791
catboost    0.8384   0.8811 0.7773
     dnn    0.8260   0.8678 0.7673
      dt    0.8137   0.8430 0.7407
      rf    0.8114   0.8690 0.7478
   ridge    0.8070   0.8573 0.7422
  logreg    0.8070   0.8573 0.7422
   lasso    0.7991   0.8524 0.7236
     knn    0.7946   0.8431 0.7313
    
Лог эксперимента дописан: checkpoints/model_comparisonres_experiments.csv

...