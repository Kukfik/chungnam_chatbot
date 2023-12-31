import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:/Users/admin/chungnam_chatbot/pytorch/data/weather.csv')

# dataset.plot(x='MinTemp', y="MaxTemp", style='o')
# plt.title('MinTemp vs MaxTemp')
# plt.xlabel("MinTemp")
# plt.ylabel('MaxTemp')
# plt.show()

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), "Predicted": y_pred.flatten()})
df

# plt.scatter(X_test, y_test, color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth = 2)
# plt.show()

print('평균제곱법:', metrics.mean_squared_error(y_test, y_pred))
print('루트', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))