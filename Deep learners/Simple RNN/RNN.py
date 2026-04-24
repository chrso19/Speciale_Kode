import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# =========================
# HARD CODED SETTINGS
# =========================
FILE_PATH = "combined_data_cleaned_v5.csv"
PRICE_ZONE = "DK1"                  # "DK1" eller "DK2"
TRAIN_HOURS = 8760             # fx 1 år = 8760 timer
TARGET_TIME = "2024-01-01 00:00:00"
SEQ_LEN = 24                       # brug de sidste 24 timer som input
EPOCHS = 20
LEARNING_RATE = 0.001

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(FILE_PATH, decimal=",")

# gør TimeUTC til datetime på samme simple måde som i jeres loader
df["Time"] = pd.to_datetime(df["TimeUTC"].str[:19])

# behold kun valgt zone
df = df[df["DKZone"] == PRICE_ZONE].copy()
df = df.sort_values("Time").reset_index(drop=True)

# find target-tidspunktet
target_time = pd.Timestamp(TARGET_TIME)

# vi vil bruge data før target_time
history = df[df["Time"] < target_time].copy()

if len(history) < TRAIN_HOURS + SEQ_LEN:
    raise ValueError(
        f"Ikke nok historik før {TARGET_TIME}. "
        f"Har {len(history)} timer, men skal bruge mindst {TRAIN_HOURS + SEQ_LEN}."
    )

# tag kun de sidste TRAIN_HOURS + SEQ_LEN timer
history = history.iloc[-(TRAIN_HOURS + SEQ_LEN):].copy()

# kun price-serien
prices = history["DKPrice"].astype(float).values.reshape(-1, 1)

# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# =========================
# MAKE SEQUENCES
# =========================
X, y = [], []
for i in range(len(prices_scaled) - SEQ_LEN):
    X.append(prices_scaled[i:i + SEQ_LEN])
    y.append(prices_scaled[i + SEQ_LEN])

X = np.array(X)   # shape: (samples, seq_len, 1)
y = np.array(y)   # shape: (samples, 1)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# =========================
# SIMPLE RNN MODEL
# =========================
class SimpleRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]   # tag sidste timestep
        out = self.fc(out)
        return out

model = SimpleRNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# =========================
# PREDICT NEXT HOUR
# =========================
# brug de sidste SEQ_LEN kendte timer før target_time
last_sequence = prices_scaled[-SEQ_LEN:]
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    pred_scaled = model(last_sequence).cpu().numpy()

pred = scaler.inverse_transform(pred_scaled)[0, 0]

print("\n=========================")
print(f"Price zone: {PRICE_ZONE}")
print(f"Predicted time: {TARGET_TIME}")
print(f"Predicted DKPrice: {pred:.2f}")
print("=========================")