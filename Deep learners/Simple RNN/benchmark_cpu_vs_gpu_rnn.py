import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

FILE_PATH = r"C:\Users\n_and\OneDrive\Delt skrivebord\Data Science\Speciale\Energinet\Delte scripts\Speciale_Kode\Data\combined_data_cleaned_v5.csv"
PRICE_ZONE = "DK1"
TARGET_COL = "DKPrice"
SEQ_LEN = 24
TRAIN_HOURS = 8760
HIDDEN_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
N_FORWARD_RUNS = 500
SEED = 42


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_series(file_path, price_zone, target_col):
    df = pd.read_csv(file_path, decimal=",")

    if "TimeUTC" in df.columns:
        df["Time"] = pd.to_datetime(df["TimeUTC"].astype(str).str[:19])
    elif "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        raise ValueError("Could not find 'TimeUTC' or 'Time' column in dataset.")

    if "DKZone" not in df.columns:
        raise ValueError("Could not find 'DKZone' column in dataset.")

    if target_col not in df.columns:
        raise ValueError(f"Could not find target column '{target_col}' in dataset.")

    zone_df = df[df["DKZone"] == price_zone].copy()
    zone_df = zone_df.sort_values("Time").reset_index(drop=True)

    if zone_df.empty:
        raise ValueError(f"No rows found for PRICE_ZONE = '{price_zone}'.")

    return zone_df[target_col].astype(float)


def make_sequences(values_2d, seq_len):
    X, y = [], []
    for i in range(len(values_2d) - seq_len):
        X.append(values_2d[i:i + seq_len])
        y.append(values_2d[i + seq_len])

    if not X:
        raise ValueError("Not enough data to create any sequences.")

    return np.array(X), np.array(y)


def prepare_data():
    print("=== Loading data ===")
    series = load_series(FILE_PATH, PRICE_ZONE, TARGET_COL)
    print(f"Loaded zone: {PRICE_ZONE}")
    print(f"Number of observations in zone: {len(series)}")

    needed = TRAIN_HOURS + SEQ_LEN
    if len(series) < needed:
        raise ValueError(f"Need at least {needed} rows, but only found {len(series)}.")

    series = series.iloc[-needed:]
    values = series.to_numpy().reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X_np, y_np = make_sequences(values_scaled, SEQ_LEN)
    print(f"X shape (numpy): {X_np.shape}")
    print(f"y shape (numpy): {y_np.shape}")
    return X_np, y_np


def benchmark_device(device, X_np, y_np):
    print(f"\n=== Benchmark on {device} ===")
    set_seed(SEED)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    model = SimpleRNNModel(input_size=1, hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model device:", next(model.parameters()).device)
    print("X device:", X.device)
    print("y device:", y.device)

    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Allocated MB before training:", torch.cuda.memory_allocated() / 1e6)
        print("Reserved MB before training:", torch.cuda.memory_reserved() / 1e6)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    final_loss = None
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start

    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        for _ in range(N_FORWARD_RUNS):
            _ = model(X)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_elapsed = time.perf_counter() - forward_start

    allocated_mb = None
    reserved_mb = None
    peak_allocated_mb = None
    if device.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        reserved_mb = torch.cuda.memory_reserved() / 1e6
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
        print("Allocated MB after training:", allocated_mb)
        print("Reserved MB after training:", reserved_mb)
        print("Peak allocated MB:", peak_allocated_mb)

    print(f"Training time ({EPOCHS} epochs): {train_elapsed:.4f} s")
    print(f"Forward time ({N_FORWARD_RUNS} runs): {forward_elapsed:.4f} s")
    print(f"Average per forward pass: {forward_elapsed / N_FORWARD_RUNS:.6f} s")
    print(f"Final loss: {final_loss:.6f}")

    return {
        "device": str(device),
        "train_time_s": train_elapsed,
        "forward_time_s": forward_elapsed,
        "avg_forward_s": forward_elapsed / N_FORWARD_RUNS,
        "final_loss": final_loss,
        "gpu_allocated_mb": allocated_mb,
        "gpu_reserved_mb": reserved_mb,
        "gpu_peak_allocated_mb": peak_allocated_mb,
    }


def main():
    print("=== CUDA status ===")
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))

    X_np, y_np = prepare_data()

    results = []
    results.append(benchmark_device(torch.device("cpu"), X_np, y_np))

    if torch.cuda.is_available():
        results.append(benchmark_device(torch.device("cuda"), X_np, y_np))
    else:
        print("\nCUDA not available, GPU benchmark skipped.")

    print("\n=== Comparison summary ===")
    for res in results:
        print(res)

    if len(results) == 2:
        cpu = results[0]
        gpu = results[1]
        print("\n=== CPU vs GPU ===")
        print(f"Training speedup (CPU/GPU): {cpu['train_time_s'] / gpu['train_time_s']:.2f}x")
        print(f"Forward speedup  (CPU/GPU): {cpu['forward_time_s'] / gpu['forward_time_s']:.2f}x")
        print(f"GPU peak allocated MB: {gpu['gpu_peak_allocated_mb']:.3f}")


if __name__ == "__main__":
    main()
