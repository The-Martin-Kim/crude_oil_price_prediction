import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Feature columns
features = ['Copper', 'Sugar', 'Natural Gas', 'Silver',
            'Platinum', 'Feeder Cattle', 'Lean Hogs', 'Cotton',
            'Live Cattle', 'Kansas Wheat']

# Load original train and test data
x_train = pd.read_csv('clean_csv/x_train.csv')
x_test = pd.read_csv('clean_csv/x_test.csv')

# Ensure 'Date' is in datetime format
x_train['Date'] = pd.to_datetime(x_train['Date'])
x_test['Date'] = pd.to_datetime(x_test['Date'])

# Load additional data (clean_kick.csv)
clean_kick = pd.read_csv('kick_data/clean_kick.csv')
clean_kick['Date'] = pd.to_datetime(clean_kick['Date'])

# Merge clean_kick 'Close' into x_train and x_test
x_train = x_train.merge(clean_kick, on='Date', how='left')
x_test = x_test.merge(clean_kick, on='Date', how='left')

# Scale data
scaler_x = MinMaxScaler()

# Select only the specified features and 'Close'
x_train_scaled = scaler_x.fit_transform(x_train[features + ['Close']].values)
x_test_scaled = scaler_x.transform(x_test[features + ['Close']].values)

# Convert scaled data back to original scale for saving
x_train_original = scaler_x.inverse_transform(x_train_scaled)
x_test_original = scaler_x.inverse_transform(x_test_scaled)

# Define column names
columns = features + ['Close']

# Convert numpy arrays to DataFrames
x_train_df = pd.DataFrame(x_train_original, columns=columns)
x_test_df = pd.DataFrame(x_test_original, columns=columns)

# Save DataFrames to CSV files
x_train_df.to_csv('processed_x_train.csv', index=False)
x_test_df.to_csv('processed_x_test.csv', index=False)

print("Processed train and test data have been saved as 'processed_x_train.csv' and 'processed_x_test.csv'.")
