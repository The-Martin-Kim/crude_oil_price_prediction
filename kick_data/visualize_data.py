import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('clean_kick.csv')

data['Date'] = pd.to_datetime(data['Date'])

data_range_1 = data[(data['Date'] >= "2018-01-02") & (data['Date'] <= "2023-06-30")]
data_range_2 = data[(data['Date'] >= "2023-07-03") & (data['Date'] <= "2024-06-28")]

scaler = MinMaxScaler()
data_range_1['Scaled Close'] = scaler.fit_transform(data_range_1['Close'].values.reshape(-1, 1))
data_range_2['Scaled Close'] = scaler.fit_transform(data_range_2['Close'].values.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(data_range_1['Date'], data_range_1['Scaled Close'], label='2018-01-02 to 2023-06-30')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Scaled Close Price')
plt.title('Scaled Close Prices (2018-01-02 to 2023-06-30)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("train.png")

plt.figure(figsize=(12, 6))
plt.plot(data_range_2['Date'], data_range_2['Scaled Close'], label='2023-07-03 to 2024-06-28')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Scaled Close Price')
plt.title('Scaled Close Prices (2023-07-03 to 2024-06-28)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("test.png")
