from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','Class']
dataset = pd.read_csv('C:/Users/admin/chungnam_chatbot/pytorch/iris.data', names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
s = StandardScaler()
X_test = s.fit_transform(X_test)

with open('pytorch/data/knn.pickle', 'br') as f:
    knn = pickle.load(f)

y_pred = knn.predict(X_test)
print('정확도: {}'.format(accuracy_score(y_test, y_pred)))