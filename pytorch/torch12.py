import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn
df = pd.read_csv('C:/Users/admin/chungnam_chatbot/pytorch/data/train.csv', index_col='PassengerId')
print(df.head())

df = df[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({"male":0, 'female':1})
df.dronpa()
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
model = sklearn.tree.DesicionTreeClassifier()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

sklearn.metrics.accuracy_score(y_test, y_predict)

print(pd.DataFrame(
    sklearn.metrics.confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'], 
    index = ['True Not Survival', 'True Survival'])
)