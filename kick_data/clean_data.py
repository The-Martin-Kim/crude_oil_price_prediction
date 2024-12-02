import pandas as pd

df = pd.read_csv("kick_data.csv")

df.drop(columns=['Adj Close', 'High', 'Low',
                 'Open', 'Volume'], index=[0, 1], inplace=True)

df.reset_index(drop=True, inplace=True)

df.rename(columns={'Price': 'Date'}, inplace=True)

df['Date'] = df['Date'].str.replace('00:00:00\+00:00', '', regex=True)

df.to_csv("clean_kick.csv", encoding='utf-8-sig', index=False)
