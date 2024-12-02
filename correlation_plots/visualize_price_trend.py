import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

x_train = pd.read_csv('../clean_csv/x_train.csv')
x_test = pd.read_csv('../clean_csv/x_test.csv')
y_train = pd.read_csv('../clean_csv/y_train.csv')
y_test = pd.read_csv('../clean_csv/y_test.csv')

scaler = MinMaxScaler()

y_train_scaled = scaler.fit_transform(y_train.iloc[:, 1].values.reshape(-1, 1)).flatten()
y_test_scaled = scaler.fit_transform(y_test.iloc[:, 1].values.reshape(-1, 1)).flatten()


def scale_and_save_grid_with_corr(data, y_scaled, filename, title_prefix):
    scaled_data = data.copy()
    numeric_columns = scaled_data.columns[1:]
    scaled_data[numeric_columns] = scaled_data[numeric_columns].astype(float)
    scaled_data[numeric_columns] = scaler.fit_transform(scaled_data[numeric_columns])

    num_commodities = len(numeric_columns)
    rows = math.ceil(num_commodities / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, commodity in enumerate(numeric_columns):
        commodity_series = scaled_data[commodity].values
        corr = pd.Series(commodity_series).corr(pd.Series(y_scaled))
        axes[i].plot(commodity_series)
        axes[i].set_title(f'{title_prefix} - {commodity} (Corr: {corr:.2f})')
        axes[i].set_ylabel('Scaled Close Price')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


scale_and_save_grid_with_corr(x_train, y_train_scaled, 'x_train_trends_with_corr.png', 'x_train (Scaled)')
scale_and_save_grid_with_corr(x_test, y_test_scaled, 'x_test_trends_with_corr.png', 'x_test (Scaled)')


def scale_and_save_single(data, y_scaled, filename, title):
    scaled_data = data.copy()
    scaled_data.iloc[:, 1] = scaled_data.iloc[:, 1].astype(float)
    scaled_data.iloc[:, 1] = scaler.fit_transform(scaled_data.iloc[:, 1].values.reshape(-1, 1))

    corr = pd.Series(scaled_data.iloc[:, 1].values).corr(pd.Series(y_scaled))

    plt.figure(figsize=(8, 4))
    plt.plot(scaled_data.iloc[:, 1], label=f'Oil (Corr: {corr:.2f})')
    plt.ylabel('Scaled Close Price')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


scale_and_save_single(y_train, y_train_scaled, 'y_train_trend_with_corr.png', 'y_train (Scaled)')
scale_and_save_single(y_test, y_test_scaled, 'y_test_trend_with_corr.png', 'y_test (Scaled)')
