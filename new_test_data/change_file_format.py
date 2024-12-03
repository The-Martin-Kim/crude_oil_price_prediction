import pandas as pd

file_path = 'test_x_data_new.xlsx'
x_train = pd.read_excel(file_path)

x_train['Date'] = pd.to_datetime(x_train['Date'])

transformed_data = x_train.copy()

output_path = 'new_x_test.csv'
transformed_data.to_csv(output_path, index=False)
