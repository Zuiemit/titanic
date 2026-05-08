from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        act_map = {
            'relu': lambda: nn.ReLU(),
            'tanh': lambda: nn.Tanh(),
            'selu': lambda: nn.SELU(),
        }
        act_fn = act_map[activation]()

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                act_fn,
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class DNNClassifier:

    def __init__(
        self,
        hidden_sizes: List[int] = (256, 128, 64),
        dropout: float = 0.3,
        activation: str = 'relu',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 15,
        scheduler: str = 'cosine',
        device: str = 'cpu',
        optimizer: str = 'AdamW'
    ) -> None:
        self.hidden_sizes  = list(hidden_sizes)
        self.dropout       = dropout
        self.activation    = activation
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.patience      = patience
        self.scheduler_type = scheduler
        self.device        = device
        self.optimizer = optimizer

        self.model_   = None
        self.scaler_  = StandardScaler()
        self.best_state_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series    = None,
    ) -> 'DNNClassifier':
        """Обучает модель. Если переданы X_val/y_val — использует early stopping."""

        # Нормализация — всегда нужна для DNN
        X_np = self.scaler_.fit_transform(X.values.astype(np.float32))
        y_np = y.values.astype(np.float32)

        X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_np, dtype=torch.float32).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Валидационные тензоры
        if X_val is not None and y_val is not None:
            X_val_np = self.scaler_.transform(X_val.values.astype(np.float32))
            X_val_t  = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
            y_val_t  = torch.tensor(y_val.values.astype(np.float32), dtype=torch.float32).to(self.device)
        else:
            X_val_t = y_val_t = None

        # Строим модель
        self.model_ = MLP(
            input_size   = X_np.shape[1],
            hidden_sizes = self.hidden_sizes,
            dropout      = self.dropout,
            activation   = self.activation,
        ).to(self.device)
        
        params = self.model_.parameters()
        lr = self.learning_rate
        wd = self.weight_decay

        if self.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )
        elif self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
        else:
            scheduler = None

        criterion = nn.BCEWithLogitsLoss()

        best_val_loss   = float('inf')
        patience_counter = 0
        self.best_state_ = None

        for epoch in range(self.epochs):

            # --- Train ---
            self.model_.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            # --- Early stopping по val loss ---
            if X_val_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_t)
                    val_loss   = criterion(val_logits, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    self.best_state_ = {
                        k: v.cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

        # Восстанавливаем лучшие веса
        if self.best_state_ is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state_.items()}
            )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает массив [n_samples, 2] как sklearn."""
        self.model_.eval()
        X_np = self.scaler_.transform(X.values.astype(np.float32))
        X_t  = torch.tensor(X_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model_(X_t).cpu().numpy()

        proba_pos = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
