import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = ['Copper', 'Sugar', 'Natural Gas', 'Silver',
            'Feeder Cattle', 'Lean Hogs', 'Cotton']

sequence_length = 20
batch_size = 32
hidden_size = 64
output_size = 1
num_layers = 3
num_epochs = 80
learning_rate = 0.0001

# Load original train and test data
x_train = pd.read_csv('clean_csv/x_train.csv')
y_train = pd.read_csv('clean_csv/y_train.csv')
x_test = pd.read_csv('clean_csv/x_test.csv')
y_test = pd.read_csv('clean_csv/y_test.csv')

# Ensure 'Date' is in datetime format
x_train['Date'] = pd.to_datetime(x_train['Date'])
x_test['Date'] = pd.to_datetime(x_test['Date'])

# Load additional data (clean_kick.csv)
clean_kick = pd.read_csv('kick_data/clean_kick.csv')
clean_kick['Date'] = pd.to_datetime(clean_kick['Date'])

# Merge clean_kick 'Close' into x_train and x_test
x_train = x_train.merge(clean_kick, on='Date', how='left')
x_test = x_test.merge(clean_kick, on='Date', how='left')

# Extract only the target column for y_train and y_test
y_train = y_train.iloc[:, 1].values  # Assuming the second column is the target
y_test = y_test.iloc[:, 1].values

# Convert y_train and y_test to float
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Scale data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Select only the specified features and 'Close'
x_train = x_train[features + ['Close']].values
x_test = x_test[features + ['Close']].values

# Scale features and targets
x_train = scaler_x.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
x_test = scaler_x.transform(x_test)
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Function to create sequences
def create_sequences(data_x, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_x) - seq_length):
        x_seq = data_x[i:i + seq_length]
        y_seq = data_y[i + seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(
        np.array(ys), dtype=torch.float32
    )

# Create sequences
x_train_seq, y_train_seq = create_sequences(x_train, y_train, sequence_length)
x_test_seq, y_test_seq = create_sequences(x_test, y_test, sequence_length)

# Create DataLoaders
train_data = TensorDataset(x_train_seq, y_train_seq)
test_data = TensorDataset(x_test_seq, y_test_seq)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# GRU model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.fc4 = nn.Linear(hidden_size // 8, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out

model = GRUModel(input_size=len(features) + 1,
                     hidden_size=hidden_size,
                     output_size=output_size,
                     num_layers=num_layers).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluate the model
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

train_rmse, train_predictions, train_actuals = calculate_rmse(model, train_loader, scaler_y)
test_rmse, test_predictions, test_actuals = calculate_rmse(model, test_loader, scaler_y)

print(f"Final Train RMSE: {train_rmse:.4f}")
print(f"Final Test RMSE: {test_rmse:.4f}")

# Plot actual vs predicted for the test set
plt.figure(figsize=(12, 6))
plt.plot(test_actuals, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted')
plt.savefig("new_result_gpu.png")
plt.show()
