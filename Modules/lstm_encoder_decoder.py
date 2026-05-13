"""
LSTM Encoder-Decoder for multivariate sequence-to-one price forecasting.

Architecture
------------
Encoder : stacked LSTM that reads the input window (seq_len, n_features)
          and produces a final hidden / cell state.
Decoder : single-step LSTM that starts from the encoder state and, at each
          decoding step, receives the previous target value (teacher-forcing
          during training, recursive during inference).  The final hidden
          state is projected to a scalar price forecast.

The public API matches TorchRNNRegressor exactly so that the same
week_predictions2.get_predictions() and Validation3.run_cross_validation()
infrastructure can be reused without any changes.

Key attributes that week_predictions2 reads from the model object:
    model.sequence_length   - encoder look-back window length
    model.use_target_history - False  (decoder handles target internally;
                                the exog window built by _build_rnn_window
                                should NOT include target columns)

Teacher-forcing modes (training_prediction param):
    'recursive'            - decoder always feeds its own previous output
    'teacher_forcing'      - decoder always receives the true target value
    'mixed_teacher_forcing'- each step randomly chooses based on
                             teacher_forcing_ratio
    dynamic_tf=True        - linearly decay teacher_forcing_ratio → 0 over epochs
"""

import copy
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
# Core PyTorch modules
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """Encodes an input sequence into a final (hidden, cell) state."""

    def __init__(self, input_size: int, hidden_size: int, layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    """
    Single-step LSTM decoder.
    Input at each step: (batch, 1, 1)  - previous target value.
    Output: scalar prediction for the current step.
    """

    def __init__(self, hidden_size: int, layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        # x: (batch, 1, 1)
        out, hidden = self.lstm(x, hidden)   # out: (batch, 1, hidden_size)
        pred = self.fc(out[:, -1, :])        # (batch, 1)
        return pred, hidden


class Seq2SeqLSTM(nn.Module):
    """
    Seq2Seq wrapper used by TorchLSTMEncoderDecoder.

    During training it can use recursive / teacher-forcing / mixed decoding.
    During inference it always decodes recursively (one step at a time) to
    produce a single scalar forecast.
    """

    def __init__(self, input_size: int, hidden_size: int, layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, layers, dropout)

    def forward(self, x_enc, target_len: int = 1,
                y_true=None,
                training_prediction: str = "recursive",
                teacher_forcing_ratio: float = 0.5):
        """
        x_enc  : (batch, seq_len, input_size)
        target_len : number of future steps to produce
        y_true : (batch, target_len) ground-truth targets, only used during
                 teacher-forcing training; may be None at inference.

        Returns outputs: (batch, target_len)
        """
        batch = x_enc.size(0)
        h, c = self.encoder(x_enc)

        decoder_input = x_enc[:, -1:, :1]
        decoder_hidden = (h, c)

        outputs = torch.zeros(batch, target_len, device=x_enc.device)

        # Ensure y_true is (B, target_len) regardless of how it arrives
        if y_true is not None and y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)   # (B,) -> (B, 1)

        for t in range(target_len):
            pred, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = pred.squeeze(-1)

            if self.training and y_true is not None:
                if training_prediction == "teacher_forcing":
                    decoder_input = y_true[:, t:t+1].unsqueeze(-1)
                elif training_prediction == "mixed_teacher_forcing":
                    use_tf = torch.rand(1).item() < teacher_forcing_ratio
                    decoder_input = (
                        y_true[:, t:t+1].unsqueeze(-1) if use_tf
                        else pred.unsqueeze(1)
                    )
                else:
                    decoder_input = pred.unsqueeze(1)
            else:
                decoder_input = pred.unsqueeze(1)

        return outputs


# ---------------------------------------------------------------------------
# Scikit-learn wrapper
# ---------------------------------------------------------------------------

class TorchLSTMEncoderDecoder(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible LSTM Encoder-Decoder.

    Designed as a drop-in replacement for TorchRNNRegressor, so it exposes
    the same attributes (`sequence_length`, `use_target_history`) and the
    same `predict(X)` signature accepting a 3-D window built by
    week_predictions2._build_rnn_window().

    Parameters
    ----------
    hidden_size : int
        Number of units in the LSTM hidden state.
    layers : int
        Number of stacked LSTM layers in both encoder and decoder.
    learning_rate : float
    epochs : int
        Maximum training epochs.
    batch_size : int
    sequence_length : int
        Encoder look-back window (hours).  Passed to _build_rnn_window via
        the `model.sequence_length` attribute.
    dropout : float
        Dropout between stacked LSTM layers (ignored for layers == 1).
    teacher_forcing_ratio : float
        Probability of using teacher forcing in 'mixed_teacher_forcing' mode.
    training_prediction : str
        One of 'recursive', 'teacher_forcing', 'mixed_teacher_forcing'.
    dynamic_tf : bool
        If True, linearly decay teacher_forcing_ratio → 0 over epochs.
    patience : int
        Early-stopping patience (epochs without improvement on train loss).
        Set to 0 to disable.
    random_state : int
    log_epoch_metrics : bool
        If True, log per-epoch metrics to an active W&B run.
    log_prefix : str
        Prefix added to W&B metric names.
    warm_start : bool
        If True, reuse existing model weights across fit() calls.
    """

    # week_predictions2 checks this to decide whether to append target history
    # to the exogenous window.  The encoder-decoder manages target info itself
    # through the decoder seed, so we set it to False.
    use_target_history: bool = False

    def __init__(
        self,
        hidden_size: int = 64,
        layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 64,
        sequence_length: int = 168,
        dropout: float = 0.0,
        teacher_forcing_ratio: float = 0.5,
        training_prediction: str = "mixed_teacher_forcing",
        dynamic_tf: bool = False,
        patience: int = 10,
        random_state: int = 42,
        log_epoch_metrics: bool = False,
        log_prefix: str = "",
        warm_start: bool = False,
    ):
        self.hidden_size = hidden_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.training_prediction = training_prediction
        self.dynamic_tf = dynamic_tf
        self.patience = patience
        self.random_state = random_state
        self.log_epoch_metrics = log_epoch_metrics
        self.log_prefix = log_prefix
        self.warm_start = warm_start

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, X: np.ndarray) -> torch.Tensor:
        """
        Convert a 2-D feature matrix (n_samples, n_features) into a 3-D
        tensor (n_samples, sequence_length, n_features) using a sliding
        window with left-padding.
        """
        n, f = X.shape
        seq = self.sequence_length
        pad = np.repeat(X[:1], seq - 1, axis=0)
        padded = np.vstack([pad, X])            # (n + seq - 1, f)
        out = np.empty((n, seq, f), dtype=np.float32)
        for i in range(n):
            out[i] = padded[i: i + seq]
        return torch.tensor(out, dtype=torch.float32)

    def _initialize(self, input_size: int):
        dev = self._device()
        self.device_ = dev
        self.pin_memory_ = dev.type == "cuda"
        self.input_size_ = input_size
        self.model_ = Seq2SeqLSTM(
            input_size=input_size,
            hidden_size=int(self.hidden_size),
            layers=int(self.layers),
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
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        set_seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)

        if X_np.ndim != 2:
            raise ValueError(
                f"fit() expects 2-D X (n_samples, n_features), got {X_np.shape}."
            )

        X_tensor = self._build_sequences(X_np)           # (n, seq, feat)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        # For teacher forcing we need y shaped (n, 1) – target_len == 1 here.

        needs_reinit = (
            not self.warm_start
            or not hasattr(self, "model_")
            or not hasattr(self, "input_size_")
            or int(self.input_size_) != X_tensor.shape[-1]
        )
        if needs_reinit:
            self._initialize(X_tensor.shape[-1])

        dev = self.device_

        dataset_local = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset_local,
            batch_size=int(self.batch_size),
            shuffle=True,
            pin_memory=self.pin_memory_,
        )

        best_loss = float("inf")
        patience_counter = 0
        tf_ratio = float(self.teacher_forcing_ratio)

        for epoch in range(int(self.epochs)):
            self.model_.train()
            batch_losses, all_preds, all_targets = [], [], []

            # Dynamic teacher forcing: linearly decay ratio to 0.
            if self.dynamic_tf and self.training_prediction in (
                "teacher_forcing", "mixed_teacher_forcing"
            ):
                tf_ratio = float(self.teacher_forcing_ratio) * (
                    1.0 - epoch / max(1, int(self.epochs) - 1)
                )

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(dev, non_blocking=self.pin_memory_)
                y_batch = y_batch.to(dev, non_blocking=self.pin_memory_)   # (B, 1)

                self.optimizer_.zero_grad()
                preds = self.model_(
                    X_batch,
                    target_len=1,
                    y_true=y_batch,
                    training_prediction=self.training_prediction,
                    teacher_forcing_ratio=tf_ratio,
                )                                          # (B, 1)
                loss = self.loss_fn_(preds.squeeze(-1), y_batch.squeeze(-1))
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

            # Early stopping on training loss.
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
        """
        Accept either:
          • 2-D  (n_samples, n_features)   – build sequences internally
          • 3-D  (n_samples, seq_len, n_features) – use as-is (from _build_rnn_window)
        Returns 1-D array of predictions, length == n_samples.
        """
        X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim == 2:
            X_tensor = self._build_sequences(X_np)
        elif X_np.ndim == 3:
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
        else:
            raise ValueError(
                f"predict() expects 2-D or 3-D X, got shape {X_np.shape}."
            )

        self.model_.eval()
        dev = self.device_
        with torch.no_grad():
            X_tensor = X_tensor.to(dev, non_blocking=self.pin_memory_)
            preds = self.model_(X_tensor, target_len=1).squeeze(-1)
        return preds.detach().cpu().numpy().reshape(-1)
