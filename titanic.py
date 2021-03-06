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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


train_csv=pd.read_csv('test.csv')
df=pd.read_csv('train.csv')
target=pd.read_csv('gender_submission.csv')

plt.hist(df['Fare'])

sns.boxplot(df['Age'])

print(df.isnull().sum())

Mrmean=[]
Mrsmean=[]
MissMean=[]
Mastermean=[]

def age_feature():
    for i in range(len(df)):
        if 'Master' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == False:
                Mastermean.append(df['Age'][i])

        elif 'Mr.' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == False:   
                Mrmean.append(df['Age'][i])

        elif 'Mrs.' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == False:
                Mrsmean.append(df['Age'][i])
            
        elif  'Miss' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == False:
                MissMean.append(df['Age'][i])
    
    for i in range(len(df)):
        if 'Master' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == True:
                df.loc[i,'Age']=(statistics.median(sorted(Mastermean)))
        elif 'Mr.' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == True:
    #            df['Age'][i]=(statistics.median(sorted(Mrmean)))
                    df.loc[i,'Age']=(statistics.median(sorted(Mrmean)))
        elif 'Mrs.' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == True:
                df.loc[i,'Age']=(statistics.median(sorted(Mrsmean)))
        elif  'Miss' in df['Name'][i]:
            if math.isnan(df['Age'][i]) == True:
                df.loc[i,'Age']=(statistics.median(sorted(MissMean)))

age_eng = age_feature()

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

dc=DecisionTreeClassifier(splitter='random',
 random_state= 10,
 min_samples_split = 5,
 min_samples_leaf = 2,
 max_depth = 10,
 criterion = 'gini')
dc.fit(dc_Xtrain,dc_ytrain)
dc_pred=dc.predict(dc_Xtest)
print(accuracy_score(dc_ytest,dc_pred))

dcparams={}
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
dcparams={'criterion':['gini','entropy'],'splitter':["best", "random"],'max_depth':max_depth,'min_samples_split':[2, 5, 10],'min_samples_leaf': [1,2,4],'random_state':[0,5,10,20,25,100]}


dc_rand=RandomizedSearchCV(dc,dcparams,cv=3,n_jobs=-1,random_state=100)
dc_rand.fit(dc_Xtrain,dc_ytrain)
dc_rand.best_params_


Ada=AdaBoostClassifier(base_estimator=dc,n_estimators=500,learning_rate=1)
boost=Ada.fit(dc_Xtrain,dc_ytrain)
boost_pred=boost.predict(dc_Xtest)
print(accuracy_score(dc_ytest,boost_pred))

dcparams={'criterion':['gini','entropy'],'splitter':["best", "random"],
'max_depth':[None,dc_rand.best_params_['max_depth']-5,dc_rand.best_params_['max_depth']-10,dc_rand.best_params_['max_depth'],dc_rand.best_params_['max_depth']+20,dc_rand.best_params_['max_depth']+50], 
'min_samples_split':[dc_rand.best_params_['min_samples_split']-1,dc_rand.best_params_['min_samples_split']-2,dc_rand.best_params_['min_samples_split'],dc_rand.best_params_['min_samples_split']+2,dc_rand.best_params_['min_samples_split']+4],
'min_samples_leaf':[dc_rand.best_params_['min_samples_leaf']-2,dc_rand.best_params_['min_samples_leaf']-4,dc_rand.best_params_['min_samples_leaf'],dc_rand.best_params_['min_samples_leaf']+2,dc_rand.best_params_['min_samples_leaf']+4],
'random_state':[0,5,10,20,25,100]}

dfgrid=GridSearchCV(dc,dcparams,cv=3,n_jobs=-1)
dfgrid.fit(dc_Xtrain,dc_ytrain)
dfgridpred=dfgrid.predict(dc_Xtest)

rf=RandomForestClassifier(n_estimators= 1400,min_samples_split= 5,min_samples_leaf= 2,max_depth= 90,criterion= 'entropy',bootstrap= True)
rf.fit(dc_Xtrain,dc_ytrain)
y_rfpred=rf.predict(dc_Xtest)
print(dc_ytest,y_rfpred)

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
rfparams={'n_estimators':[int(x) for x in np.linspace(100,1500,15)],'criterion':['gini','entropy'],'max_depth':max_depth,'min_samples_split':[2, 5, 10],'bootstrap':[True,False],'min_samples_leaf': [1,2,4]}
rfrnd=RandomizedSearchCV(rf,rfparams,cv=5,random_state=40,n_jobs=-1,n_iter=100)
rfrnd.fit(dc_Xtrain,dc_ytrain)
rfrnd.best_params_
gdparams={'n_estimators':[rfrnd.best_params_['n_estimators']-100,rfrnd.best_params_['n_estimators']-200,rfrnd.best_params_['n_estimators'],rfrnd.best_params_['n_estimators']+100,rfrnd.best_params_['n_estimators']+200,rfrnd.best_params_['n_estimators']+300],
        'criterion':[rfrnd.best_params_['criterion']],
        'max_depth':[None,rfrnd.best_params_['max_depth']-10,rfrnd.best_params_['max_depth'],rfrnd.best_params_['max_depth']-20,rfrnd.best_params_['max_depth'],rfrnd.best_params_['max_depth']+10,rfrnd.best_params_['max_depth']+20],
        'min_samples_split':[rfrnd.best_params_['min_samples_split']-10,rfrnd.best_params_['min_samples_split']-20,rfrnd.best_params_['min_samples_split'],rfrnd.best_params_['min_samples_split']+10,rfrnd.best_params_['min_samples_split']+20],'bootstrap':[True,False],'min_samples_leaf': [1,2,3]}
rggrid=GridSearchCV(rf,gdparams,cv=3,n_jobs=-1)
rggrid.fit(dc_Xtrain,dc_ytrain)
ygridpred=rggrid.best_estimator_
new_y=ygridpred.predict(dc_Xtest)
print(accuracy_score(dc_ytest,new_y))

xgbm=XGBClassifier( )
xgbm.fit(X_train,y_train)
xgb_pred=xgbm.predict(X_test)
print(accuracy_score(y_test,xgb_pred))

rand_params={'learning_rate':[0.1,0.01,0.001,0.001],'max_depth':[1,3,5,7,10],'n_estimators':[50,100,200,300,500],
            "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7]}
cgb_rand=RandomizedSearchCV(xgbm,rand_params,cv=3,n_jobs=-1,random_state=40)
cgb_rand.fit(X_train,y_train)
print(cgb_rand.best_params_   )            
grid_params={'learning_rate':[cgb_rand.best_params_['learning_rate'],cgb_rand.best_params_['learning_rate']+0.001,cgb_rand.best_params_['learning_rate']+0.03],
             'max_depth':[cgb_rand.best_params_['max_depth']-1,cgb_rand.best_params_['max_depth'],cgb_rand.best_params_['max_depth']+1,cgb_rand.best_params_['max_depth']+2],
             'n_estimators':[cgb_rand.best_params_['n_estimators']-20,cgb_rand.best_params_['n_estimators'],cgb_rand.best_params_['n_estimators']+20,cgb_rand.best_params_['n_estimators']+30,cgb_rand.best_params_['n_estimators']+100],
            "gamma" : [0,cgb_rand.best_params_['gamma']+0.1,cgb_rand.best_params_['gamma']+0.001,cgb_rand.best_params_['gamma']+0.2,cgb_rand.best_params_['gamma']+0.3,cgb_rand.best_params_['gamma']+0.4],
               "colsample_bytree": [cgb_rand.best_params_['colsample_bytree']-0.5,cgb_rand.best_params_['colsample_bytree']-0.3,cgb_rand.best_params_['colsample_bytree'],cgb_rand.best_params_['colsample_bytree']+0.1,cgb_rand.best_params_['colsample_bytree']+0.001],
               "subsample": [cgb_rand.best_params_['subsample']-0.2,cgb_rand.best_params_['subsample']-0.3,cgb_rand.best_params_['subsample'],cgb_rand.best_params_['subsample']+0.1,cgb_rand.best_params_['subsample']+0.001],
               "reg_alpha": [cgb_rand.best_params_['reg_alpha']+0.3,cgb_rand.best_params_['reg_alpha']+0.4,cgb_rand.best_params_['reg_alpha'],cgb_rand.best_params_['reg_alpha']+0.8],
               "reg_lambda": [0,cgb_rand.best_params_['reg_lambda']+1,cgb_rand.best_params_['reg_lambda']+0.7,cgb_rand.best_params_['reg_lambda'],cgb_rand.best_params_['reg_lambda']+4.5,cgb_rand.best_params_['reg_lambda']+2.5],
               "min_child_weight": [cgb_rand.best_params_['min_child_weight']+30,cgb_rand.best_params_['min_child_weight']+5,cgb_rand.best_params_['min_child_weight'],cgb_rand.best_params_['min_child_weight']+10,cgb_rand.best_params_['min_child_weight']+20]}
grid_cgb=GridSearchCV(xgbm,grid_params,cv=10,n_jobs=1)


new_tr=pd.get_dummies(train_csv['Sex'],drop_first=True)
filt=train_csv.drop(['Parch','SibSp','PassengerId','Cabin','Name','Sex','Embarked','Ticket'],axis=1)
train_x=pd.concat([filt,new_tr],axis=1)
train_x['Fare'].fillna(train_x['Fare'].mean(),inplace=True)
train_x['Age'].fillna(train_x['Age'].mean(),inplace=True)
tr_X=train_x.iloc[:,:]
tr_X_sc=sc.fit_transform(tr_X)
y_xgb_pred=xgbm.predict(tr_X_sc).astype(int)
tr_X['Survived']=pd.DataFrame({'Survived':y_xgb_pred})
new_dat=pd.concat([train_csv['PassengerId'],tr_X['Survived']],axis=1)
new_dat.to_csv('submission.csv',index=False)
submitt_csv=pd.read_csv('submission.csv')




