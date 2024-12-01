import pandas as pd
import yfinance as yf

start = "2008-01-01"
end = "2024-07-01"

symbol = "^TNX"

df = yf.download(symbol, start, end)

df.to_csv("kick_data.csv")