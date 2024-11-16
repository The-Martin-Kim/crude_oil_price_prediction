import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_train = pd.read_csv('clean_csv/x_train.csv').drop(columns=['Date'])
y_train = pd.read_csv('clean_csv/y_train.csv').iloc[:, 1]

model = LinearRegression()
model.fit(x_train, y_train)

coefficients = pd.Series(model.coef_, index=x_train.columns)
coefficients = coefficients.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10, 6))
coefficients.plot(kind='bar', color=['royalblue' if c > 0 else 'salmon' for c in coefficients])
plt.title('Feature Importance based on Regression Coefficients (Train)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('train_coeffs.png')
