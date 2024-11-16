import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

features = ['Copper', 'Sugar', 'Natural Gas', 'Silver',
            'Live Cattle', 'Feeder Cattle', 'Cotton']
sequence_length = 30
batch_size = 16
hidden_size = 128
output_size = 1
num_layers = 3
num_epochs = 100
learning_rate = 0.001
dropout_rate = 0

x_train = pd.read_csv('clean_csv/x_train.csv').set_index('Date')
y_train = pd.read_csv('clean_csv/y_train.csv').set_index('Date').iloc[:, 0:1]
x_test = pd.read_csv('clean_csv/x_test.csv').set_index('Date')
y_test = pd.read_csv('clean_csv/y_test.csv').set_index('Date').iloc[:, 0:1]

x_train = x_train[features].values
y_train = y_train.values
x_test = x_test[features].values
y_test = y_test.values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train)
x_test = scaler_x.transform(x_test)
y_test = scaler_y.transform(y_test)


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


x_train_seq, y_train_seq = create_sequences(x_train, y_train, sequence_length)
x_test_seq, y_test_seq = create_sequences(x_test, y_test, sequence_length)

train_data = TensorDataset(x_train_seq, y_train_seq)
test_data = TensorDataset(x_test_seq, y_test_seq)
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


class GRUModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, dropout_rate
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


model = GRUModel(
    input_size=len(features),
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def calculate_rmse(model, data_loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            pred = model(x_batch)
            predictions.append(pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    return rmse, predictions, actuals


train_rmse, train_predictions, train_actuals = calculate_rmse(
    model, train_loader, scaler_y
)
test_rmse, test_predictions, test_actuals = calculate_rmse(
    model, test_loader, scaler_y
)

print(f"Final Train RMSE: {train_rmse:.4f}")
print(f"Final Test RMSE: {test_rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(test_actuals, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted')
plt.savefig("result.png")
