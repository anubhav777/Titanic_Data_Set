import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

train_csv=pd.read_csv('test.csv')
df=pd.read_csv('train.csv')
target=pd.read_csv('gender_submission.csv')

plt.hist(df['Fare'])

sns.boxplot(df['Age'])

print(df.isnull().sum())

new_data = df.drop(labels=['Sex','Name'],axis=1)

gh = pd.get_dummies(df['Sex'],drop_first=True)
new_data = df.drop(labels=['Sex','Name','Ticket','Embarked','Cabin'],axis=1)
df = pd.concat([new_data,gh],axis=1)

corr_data=df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_data,cmap='coolwarm',annot=True)

df=df.dropna()
X=df.drop(labels='Survived',axis=1).values


y=df['Survived'].values

df=df.dropna()
X=df.drop(labels=['Survived','Parch','SibSp','PassengerId'],axis=1).values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=44)

reg=LogisticRegression()

dc_Xtrain=X_train
dc_Xtest=X_test
dc_ytrain=y_train
dc_ytest=y_test

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

print(accuracy_score(y_test,y_pred))

