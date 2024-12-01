import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = [
    'Copper', 'Sugar', 'Natural Gas', 'Silver',
    'Feeder Cattle', 'Lean Hogs', 'Cotton',
    'Live Cattle', 'Kansas Wheat', 'Platinum'
]

sequence_lengths = [30, 40]
batch_sizes = [16]
hidden_sizes = [16]
num_layers_list = [2, 3]
num_epochs_list = [70, 80, 90, 100]
learning_rates = [0.00001, 0.00005]

if not os.path.exists("results"):
    os.makedirs("results")

x_train = pd.read_csv('clean_csv/x_train.csv').set_index('Date')
y_train = pd.read_csv('clean_csv/y_train.csv').set_index('Date').iloc[:, 0:1]
x_test = pd.read_csv('clean_csv/x_test.csv').set_index('Date')
y_test = pd.read_csv('clean_csv/y_test.csv').set_index('Date').iloc[:, 0:1]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train_values = scaler_x.fit_transform(x_train[features].values)
y_train_values = scaler_y.fit_transform(y_train.values)
x_test_values = scaler_x.transform(x_test[features].values)
y_test_values = scaler_y.transform(y_test.values)


def create_sequences(data_x, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_x) - seq_length):
        x_seq = data_x[i:i + seq_length]
        y_seq = data_y[i + seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def calculate_rmse(model, data_loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            predictions.append(pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    return rmse, predictions, actuals


results = []

param_combinations = list(product(
    sequence_lengths, batch_sizes, hidden_sizes, num_layers_list, num_epochs_list, learning_rates
))

for i, (
    seq_len, batch_size, hidden_size, num_layers, num_epochs, lr
) in enumerate(tqdm(param_combinations, desc="Grid Search Progress", leave=True, position=0), start=1):
    x_train_seq, y_train_seq = create_sequences(x_train_values, y_train_values, seq_len)
    x_test_seq, y_test_seq = create_sequences(x_test_values, y_test_values, seq_len)
    train_data = TensorDataset(x_train_seq, y_train_seq)
    test_data = TensorDataset(x_test_seq, y_test_seq)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = GRUModel(
        input_size=len(features), hidden_size=hidden_size, output_size=1, num_layers=num_layers
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
    train_rmse, train_predictions, train_actuals = calculate_rmse(model, train_loader, scaler_y)
    test_rmse, test_predictions, test_actuals = calculate_rmse(model, test_loader, scaler_y)
    graph_path = (
        f"results/graph_seq{seq_len}_bs{batch_size}_hs{hidden_size}_nl{num_layers}_lr{lr}.png"
    )
    plt.figure(figsize=(12, 6))
    plt.plot(test_actuals, label='Actual')
    plt.plot(test_predictions, label='Predicted')
    plt.legend()
    plt.title(f'Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
    plt.savefig(graph_path)
    plt.close()
    results.append({
        'sequence_length': seq_len,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/hyperparameter_results.csv", index=False)
