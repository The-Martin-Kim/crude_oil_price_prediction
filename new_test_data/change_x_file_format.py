import pandas as pd

x_train = pd.read_excel('test_x_data_new.xlsx')

columns = x_train.columns
unique_commodities = []
group_size = 3

for i in range(0, len(columns), group_size):
    commodity = x_train.iloc[0, i]
    if commodity not in unique_commodities:
        unique_commodities.append(commodity)

transformed_data = pd.DataFrame()

for idx, commodity in enumerate(unique_commodities):
    date_col = columns[idx * group_size + 1]
    close_col = columns[idx * group_size + 2]

    temp_df = x_train[[date_col, close_col]].copy()
    temp_df.columns = ["Date", commodity]

    if transformed_data.empty:
        transformed_data = temp_df
    else:
        transformed_data = pd.merge(transformed_data, temp_df, on="Date", how="outer")

transformed_data.to_csv('new_x_test.csv', index=False)
