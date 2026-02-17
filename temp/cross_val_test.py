# Importing packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime as dt
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os

# Load data (your existing code)
file_dir = os.path.dirname(os.path.realpath('cross_val_test.py'))
python_dir = os.path.dirname(file_dir)
speciale_path = os.path.join(python_dir, 'Speciale_Kode')
temp_path = os.path.join(speciale_path, 'temp')
data_path = os.path.join(temp_path, 'GenerationProdTypeExchange.csv')

df = pd.read_csv(data_path, sep=';', decimal=',')
df_filtered = df[df['PriceArea'] == 'DK1']
df_DK1 = df_filtered[["TimeDK", "GrossCon"]]
df_DK1 = df_DK1.sort_values(by='TimeDK', ascending=True)
df_DK1.reset_index(drop=True, inplace=True)
df_DK1['TimeDK'] = pd.to_datetime(df_DK1['TimeDK'])
df_DK1['GrossCon'] = df_DK1['GrossCon'].astype(float)

train_set = df_DK1.loc[df_DK1['TimeDK'] < pd.Timestamp('2026-02-07')]
train_set.reset_index(drop=True, inplace=True)
test_set = df_DK1.loc[df_DK1['TimeDK'] >= pd.Timestamp('2026-02-07')]
test_set.reset_index(drop=True, inplace=True)

print(f"Training data shape: {train_set.shape}")
print(f"Test data shape: {test_set.shape}")
print(f"Test set fraction: {len(test_set)/len(train_set):.2%}")

np.random.seed(42)
torch.manual_seed(42)

# ===== LSTM-SPECIFIC DATASET CLASS =====
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length=24):
        """
        data: numpy array of shape (n_samples,)
        sequence_length: how many past timesteps to use
        """
        self.sequence_length = sequence_length
        self.data = torch.FloatTensor(data).unsqueeze(1)  # Add feature dimension
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of past values
        x = self.data[idx:idx + self.sequence_length]
        # Target is the next value
        y = self.data[idx + self.sequence_length]
        return x, y

# ===== LSTM MODEL =====
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

# ===== WALK-FORWARD VALIDATION FOR PYTORCH LSTM =====
def perform_walk_forward_validation_lstm(
    data_series, 
    training_window=168,  # 7 days of hourly data
    forecast_horizon=24,  # 1 day
    sequence_length=24,   # Use past 24 hours
    hidden_size=64,
    num_layers=2,
    learning_rate=0.001,
    num_epochs=50,
    batch_size=16
):
    """
    Walk-forward validation for PyTorch LSTM.
    """
    all_predictions = []
    all_actuals = []
    fold_errors = []
    
    # Calculate number of folds
    num_folds = (len(data_series) - training_window - forecast_horizon) // forecast_horizon + 1
    
    print(f"Total folds: {num_folds}")
    print(f"Training window: {training_window} hours")
    print(f"Forecast horizon: {forecast_horizon} hours")
    print(f"Sequence length: {sequence_length} hours\n")
    
    for fold in range(num_folds):
        start = fold * forecast_horizon
        
        # Check if we have enough data
        if start + training_window + forecast_horizon > len(data_series):
            break
            
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Split data
        train_data = data_series[start:start + training_window]
        val_data = data_series[start + training_window:start + training_window + forecast_horizon]
        
        # Standardize using training statistics
        train_mean = train_data.mean()
        train_std = train_data.std()
        
        train_data_scaled = (train_data - train_mean) / train_std
        val_data_scaled = (val_data - train_mean) / train_std
        
        # Create datasets
        train_dataset = SequenceDataset(train_data_scaled, sequence_length)
        val_dataset = SequenceDataset(val_data_scaled, sequence_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model for this fold
        model = LSTMModel(
            input_size=1,  # Single feature (GrossCon)
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Make predictions
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(y_batch.squeeze().cpu().numpy())
        
        # Un-standardize
        predictions = np.array(predictions) * train_std + train_mean
        actuals = np.array(actuals) * train_std + train_mean
        
        # Calculate error
        fold_mse = mean_squared_error(actuals, predictions)
        fold_rmse = np.sqrt(fold_mse)
        fold_mae = mean_absolute_error(actuals, predictions)
        
        fold_errors.append(fold_rmse)
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
        
        print(f"  Fold RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}\n")
    
    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    overall_mae = mean_absolute_error(all_actuals, all_predictions)
    
    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall MAE: {overall_mae:.2f}")
    print(f"Average Fold RMSE: {np.mean(fold_errors):.2f} (+/- {np.std(fold_errors):.2f})")
    
    return all_predictions, all_actuals, fold_errors

# ===== PLOTTING FUNCTION =====
def plot_walk_forward_results(predictions, actuals, training_window=168, forecast_horizon=24):
    """
    Plot walk-forward validation results.
    """
    plt.figure(figsize=(15, 6))
    
    # Create x-axis (fold numbers repeated for each prediction in that fold)
    x = []
    fold_num = 0
    for i in range(len(predictions)):
        if i % (forecast_horizon) == 0 and i > 0:
            fold_num += 1
        x.append(fold_num)
    
    plt.plot(actuals, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions, label='Predicted', alpha=0.7, linewidth=2)
    
    # Add vertical lines to separate folds
    for i in range(0, len(predictions), forecast_horizon):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Time Step')
    plt.ylabel('Gross Consumption')
    plt.title('Walk-Forward Validation: Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===== RUN WALK-FORWARD VALIDATION =====
predictions, actuals, fold_errors = perform_walk_forward_validation_lstm(
    data_series=train_set['GrossCon'].values,
    training_window=168,   # 7 days
    forecast_horizon=24,   # 1 day
    sequence_length=24,    # Use past 24 hours
    hidden_size=32,        # Smaller model for limited data
    num_layers=1,          # Single layer to avoid overfitting
    learning_rate=0.001,
    num_epochs=30,         # Fewer epochs for faster iteration
    batch_size=16
)

# Plot results
plot_walk_forward_results(predictions, actuals)