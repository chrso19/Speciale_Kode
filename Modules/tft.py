"""
Temporal Fusion Transformer (TFT) for multivariate electricity-price forecasting.

Architecture
------------
This is a faithful but self-contained re-implementation of the TFT described in
    Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting".

Key components
~~~~~~~~~~~~~~
  Variable Selection Network (VSN) – learns per-feature importance weights.
  Gated Residual Network (GRN)     – the basic learnable unit, used throughout.
  Local temporal processing        – LSTM over the encoder window.
  Static enrichment + self-attention – multi-head attention over temporal features.
  Gated feed-forward output layer  – final point-forecast projection.

How it plugs into your codebase
--------------------------------
The wrapper TorchTFTRegressor exposes exactly the same interface as
TorchRNNRegressor / TorchLSTMEncoderDecoder:
    model.sequence_length    – look-back window consumed by week_predictions2
    model.use_target_history – False (TFT manages temporal context internally)
    model.fit(X, y)          – sklearn-compatible training
    model.predict(X)         – accepts 2-D or pre-built 3-D windows

Feature categorisation
~~~~~~~~~~~~~~~~~~~~~~~
The TFT distinguishes *observed* covariates (only available in the past) from
*known* covariates (available in the future, e.g. calendar/weather/capacity).
For simplicity this wrapper treats all features uniformly as observed inputs to
the encoder; the self-attention layer then lets the model learn which lags to
attend to.  If you want to pass known-future features as decoder inputs (proper
TFT style), subclass and override `_split_known_observed`.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smape_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    vals = np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    """GLU: element-wise gating on the second half of the projection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        proj = self.linear(x)
        v, g = proj.chunk(2, dim=-1)
        return v * torch.sigmoid(g)


class GatedResidualNetwork(nn.Module):
    """
    GRN(x, c=None) = LayerNorm( x + GLU( Linear( ELU( Linear(x, c) ) ) ) )

    Parameters
    ----------
    d_model   : embedding / hidden dimension
    d_context : dimension of optional static context c.  0 means no context.
    dropout   : dropout after the ELU layer
    """

    def __init__(self, d_model: int, d_context: int = 0, dropout: float = 0.0):
        super().__init__()
        self.d_context = d_context
        self.fc1 = nn.Linear(d_model + (d_context if d_context else 0), d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.glu = GatedLinearUnit(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c=None):
        if self.d_context and c is not None:
            h = self.fc1(torch.cat([x, c], dim=-1))
        else:
            h = self.fc1(x)
        h = self.dropout(torch.relu(h))
        h = self.fc2(h)
        h = self.glu(h)
        return self.norm(x + h)


class VariableSelectionNetwork(nn.Module):
    """
    Learns soft weights over features and produces a single vector per time-step.

    Each of the `n_features` raw features is projected to `d_model` via its own
    linear layer, passed through a GRN, then combined by a soft-max over a
    shared GRN that scores all features jointly.
    """

    def __init__(self, n_features: int, d_model: int,
                 d_context: int = 0, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature input projections.
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )

        # Per-feature GRNs.
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, dropout=dropout) for _ in range(n_features)]
        )

        # Flattened-input GRN that outputs selection weights.
        self.selection_grn = GatedResidualNetwork(
            n_features * d_model, d_context=d_context, dropout=dropout
        )
        self.selection_proj = nn.Linear(n_features * d_model, n_features)

    def forward(self, x, context=None):
        """
        x       : (..., n_features)   – raw feature values
        context : (..., d_context)    – optional static context
        Returns : (..., d_model)
        """
        # Project each feature independently.
        feat_embeddings = []
        for i, (proj, grn) in enumerate(
            zip(self.feature_projections, self.feature_grns)
        ):
            xi = x[..., i:i+1]           # (..., 1)
            feat_embeddings.append(grn(proj(xi)))

        # Stack: (..., n_features, d_model)
        stacked = torch.stack(feat_embeddings, dim=-2)

        # Compute selection weights.
        flat = stacked.flatten(start_dim=-2)      # (..., n_features * d_model)
        scores = self.selection_proj(
            self.selection_grn(flat, context)
        )                                          # (..., n_features)
        weights = torch.softmax(scores, dim=-1)    # (..., n_features)

        # Weighted sum.
        out = (stacked * weights.unsqueeze(-1)).sum(dim=-2)   # (..., d_model)
        return out, weights


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention with interpretable attention scores."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Causal mask so each position only attends to past.
        seq = x.size(1)
        mask = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# Full TFT module
# ---------------------------------------------------------------------------

class TFTModule(nn.Module):
    """
    Simplified TFT: all features treated as observed (encoder-only).
    Produces a single scalar forecast for the last time-step.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Variable selection (encoder) ---
        self.encoder_vsn = VariableSelectionNetwork(
            n_features=n_features,
            d_model=d_model,
            dropout=dropout,
        )

        # --- Temporal LSTM encoder ---
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # --- Static enrichment (simple GRN applied after LSTM) ---
        self.enrichment_grn = GatedResidualNetwork(d_model, dropout=dropout)

        # --- Temporal self-attention ---
        self.attn = TemporalSelfAttention(d_model, n_heads, dropout)

        # --- Position-wise feed-forward ---
        self.ff_grn = GatedResidualNetwork(d_model, dropout=dropout)

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x : (batch, seq_len, n_features)
        Returns : (batch,)  – single-step point forecast
        """
        batch, seq, _ = x.shape

        # Variable selection at each time-step.
        vsn_out, _ = self.encoder_vsn(x)             # (batch, seq, d_model)

        # LSTM temporal encoding.
        lstm_out, _ = self.lstm_encoder(vsn_out)      # (batch, seq, d_model)
        lstm_out = self.lstm_norm(vsn_out + lstm_out) # skip connection

        # Static enrichment (GRN over each time-step independently).
        enriched = self.enrichment_grn(lstm_out)      # (batch, seq, d_model)

        # Temporal self-attention.
        attn_out = self.attn(enriched)                # (batch, seq, d_model)

        # Position-wise GRN.
        ff_out = self.ff_grn(attn_out)                # (batch, seq, d_model)

        # Forecast from last time-step.
        last = ff_out[:, -1, :]                       # (batch, d_model)
        return self.output_proj(last).squeeze(-1)      # (batch,)


# ---------------------------------------------------------------------------
# Scikit-learn wrapper
# ---------------------------------------------------------------------------

class TorchTFTRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible Temporal Fusion Transformer regressor.

    Drop-in replacement for TorchRNNRegressor / TorchLSTMEncoderDecoder.
    week_predictions2.get_predictions() and Validation3.run_cross_validation()
    work without modification.

    Parameters
    ----------
    d_model : int
        Internal embedding dimension.  Must be divisible by n_heads.
    n_heads : int
        Number of attention heads.
    lstm_layers : int
        Number of LSTM layers inside TFT.
    learning_rate : float
    epochs : int
    batch_size : int
    sequence_length : int
        Encoder look-back window in hours.
    dropout : float
    patience : int
        Early-stopping patience (epochs). 0 to disable.
    random_state : int
    log_epoch_metrics : bool
    log_prefix : str
    warm_start : bool
    """

    use_target_history: bool = False   # week_predictions2 reads this

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 64,
        sequence_length: int = 168,
        dropout: float = 0.1,
        patience: int = 10,
        random_state: int = 42,
        log_epoch_metrics: bool = False,
        log_prefix: str = "",
        warm_start: bool = False,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state
        self.log_epoch_metrics = log_epoch_metrics
        self.log_prefix = log_prefix
        self.warm_start = warm_start

    # ------------------------------------------------------------------

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, X: np.ndarray) -> torch.Tensor:
        n, f = X.shape
        seq = self.sequence_length
        pad = np.repeat(X[:1], seq - 1, axis=0)
        padded = np.vstack([pad, X])
        out = np.empty((n, seq, f), dtype=np.float32)
        for i in range(n):
            out[i] = padded[i: i + seq]
        return torch.tensor(out, dtype=torch.float32)

    def _initialize(self, n_features: int):
        # Enforce d_model divisibility.
        d = self.d_model
        h = self.n_heads
        if d % h != 0:
            raise ValueError(
                f"d_model ({d}) must be divisible by n_heads ({h})."
            )
        dev = self._device()
        self.device_ = dev
        self.pin_memory_ = dev.type == "cuda"
        self.n_features_ = n_features
        self.model_ = TFTModule(
            n_features=n_features,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            lstm_layers=int(self.lstm_layers),
            dropout=float(self.dropout),
        ).to(dev)
        self.loss_fn_ = nn.MSELoss()
        self.optimizer_ = torch.optim.Adam(
            self.model_.parameters(), lr=float(self.learning_rate)
        )
        self.epoch_losses_: list = []
        self.epoch_smapes_: list = []
        self.epoch_maes_: list = []
        self.epoch_rmses_: list = []
        self._epochs_trained_: int = 0

    # ------------------------------------------------------------------

    def fit(self, X, y):
        set_seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)

        if X_np.ndim != 2:
            raise ValueError(
                f"fit() expects 2-D X (n_samples, n_features), got {X_np.shape}."
            )

        X_tensor = self._build_sequences(X_np)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        needs_reinit = (
            not self.warm_start
            or not hasattr(self, "model_")
            or not hasattr(self, "n_features_")
            or int(self.n_features_) != X_tensor.shape[-1]
        )
        if needs_reinit:
            self._initialize(X_tensor.shape[-1])

        dev = self.device_
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=int(self.batch_size),
            shuffle=True,
            pin_memory=self.pin_memory_,
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(int(self.epochs)):
            self.model_.train()
            batch_losses, all_preds, all_targets = [], [], []

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(dev, non_blocking=self.pin_memory_)
                y_batch = y_batch.to(dev, non_blocking=self.pin_memory_)

                self.optimizer_.zero_grad()
                preds = self.model_(X_batch)             # (B,)
                loss = self.loss_fn_(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                self.optimizer_.step()

                batch_losses.append(float(loss.item()))
                all_preds.append(preds.detach().cpu().numpy().reshape(-1))
                all_targets.append(y_batch.detach().cpu().numpy().reshape(-1))

            epoch_loss = float(np.mean(batch_losses))
            self.epoch_losses_.append(epoch_loss)
            self._epochs_trained_ += 1

            y_pred_ep = np.concatenate(all_preds)
            y_true_ep = np.concatenate(all_targets)
            epoch_smape = smape_mean(y_true_ep, y_pred_ep)
            epoch_mae = float(np.mean(np.abs(y_true_ep - y_pred_ep)))
            epoch_rmse = float(np.sqrt(np.mean((y_true_ep - y_pred_ep) ** 2)))

            self.epoch_smapes_.append(epoch_smape)
            self.epoch_maes_.append(epoch_mae)
            self.epoch_rmses_.append(epoch_rmse)

            if self.log_epoch_metrics:
                try:
                    import wandb
                    if wandb.run is not None:
                        p = self.log_prefix
                        wandb.log({
                            f"{p}train_MSE_loss": epoch_loss,
                            f"{p}train_smape":    epoch_smape,
                            f"{p}train_mae":      epoch_mae,
                            f"{p}train_rmse":     epoch_rmse,
                            f"{p}epoch":          int(self._epochs_trained_),
                        })
                except Exception:
                    pass

            if self.patience and self.patience > 0:
                if epoch_loss < best_loss - 1e-6:
                    best_loss = epoch_loss
                    patience_counter = 0
                    self.best_model_state_ = copy.deepcopy(self.model_.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= int(self.patience):
                        if hasattr(self, "best_model_state_"):
                            self.model_.load_state_dict(self.best_model_state_)
                        break

        return self

    def predict(self, X) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim == 2:
            X_tensor = self._build_sequences(X_np)
        elif X_np.ndim == 3:
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
        else:
            raise ValueError(
                f"predict() expects 2-D or 3-D X, got {X_np.shape}."
            )

        self.model_.eval()
        dev = self.device_
        with torch.no_grad():
            X_tensor = X_tensor.to(dev, non_blocking=self.pin_memory_)
            preds = self.model_(X_tensor)
        return preds.detach().cpu().numpy().reshape(-1)
