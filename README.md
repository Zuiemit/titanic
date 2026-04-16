# Titanic — Kaggle Competition Solution

Задача — бинарная классификация: предсказать выжил ли пассажир (`Survived = 1`) или нет (`Survived = 0`).

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

## Предобработка данных

Реализована в `dataset.py`:

---

## Валидация

Используется `StratifiedKFold` с `n_splits=5` — сохраняет баланс классов в каждом фолде.

Метрики по каждому фолду:
- `accuracy`
- `roc_auc` 
- `f1`

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