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
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

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
mms= MinMaxScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

print(accuracy_score(y_test,y_pred))

params=dict(C = uniform(loc=0, scale=4),penalty=['l1', 'l2', 'none'])

rf_randomcv=RandomizedSearchCV(reg,params,n_iter=100,n_jobs=-1,cv=5,verbose=True,random_state=100)

rf_randomcv.fit(X_train,y_train)

rf_randomcv.best_params_['penalty']

param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1','l2','none']}
grid_param=GridSearchCV(reg,param_grid,cv=3,n_jobs=-1,verbose=True)
grid_param.fit(X_train,y_train)

bst_grid=grid_param.best_estimator_

y_pr=bst_grid.predict(X_test)
print(accuracy_score(y_test,y_pr))

from sklearn.neighbors import KNeighborsClassifier

kn=KNeighborsClassifier()
kn.fit(X_train,y_train)
k_pred=kn.predict(X_test)

print(accuracy_score(y_test,k_pred))

k_params=dict(n_neighbors=[int(x) for x in np.linspace(start = 1, stop = len(X_train)/3, num = 5)],weights=['uniform','distance'],leaf_size=[int(x) for x in np.linspace(10, 100, num = 5)],algorithm  = ["auto","ball_tree","kd_tree","brute"],p=[1,2])

krand=RandomizedSearchCV(kn,k_params,cv=3,verbose=True,n_iter=100,random_state=100,n_jobs=-1)

krand.fit(X_train,y_train)
krand.best_params_['n_neighbors']
gr_params={'n_neighbors':[krand.best_params_['n_neighbors']-50,krand.best_params_['n_neighbors']-25,krand.best_params_['n_neighbors'],krand.best_params_['n_neighbors']+25,krand.best_params_['n_neighbors']+50],'p':[2],'algorithm':[krand.best_params_['algorithm']],'leaf_size':[krand.best_params_['leaf_size']-10,krand.best_params_['leaf_size']-20,krand.best_params_['leaf_size'],krand.best_params_['leaf_size']+20,krand.best_params_['leaf_size']+10]}
kdgrid=GridSearchCV(kn,gr_params,cv=3,n_jobs=-1)

kdgrid.fit(X_train,y_train)
kdbestgr=kdgrid.best_estimator_
y_kdpred=kdbestgr.predict(X_test)
print(accuracy_score(y_test,y_kdpred))