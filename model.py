# -*- coding: utf-8 -*-
"""
Predict survival on the Titanic and get familiar with ML basics
https://www.kaggle.com/c/titanic

Variable	   Definition	                                 Key

survival	   Survival	                                    0 = No, 1 = Yes
pclass	   Ticket class	                              1 = 1st, 2 = 2nd, 3 = 3rd
sex	      Sex	
Age	      Age in years	
sibsp	      # of siblings / spouses aboard the Titanic	
parch	      # of parents / children aboard the Titanic	
ticket	   Ticket number	
fare	      Passenger fare	
cabin	      Cabin number	
embarked	   Port of Embarkation	                        C = Cherbourg, Q = Queenstown, S = Southampton
"""

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

# 读入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv') 
PassengerId = test['PassengerId'] #保存用于测试的乘客ID

#解析乘客的title,如Mr,Mrs等
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
def feature_reshape(dataset):
    #dataset['Name_length'] = dataset['Name'].apply(len) #乘客名字长度
    dataset['Surname']= dataset['Name'].apply(lambda x: str(x).split('.')[1].split(' ')[1])
    dataset['Surname'] = dataset.Surname.str.replace('(', '')
    dataset['SurnameLen'] = dataset['Surname'].apply(lambda x: len(x))
    #dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1) #是否有舱位
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #家庭人数
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1 #是否是单独一人
    dataset['Embarked'] = dataset['Embarked'].fillna('S') #登船地点填补缺失值为S
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) #乘客费用填补缺失值为中位数
    age_avg = dataset['Age'].mean() #计算乘客的平均年龄
    age_std = dataset['Age'].std()  #计算乘客年龄的标准差
    age_null_count = dataset['Age'].isnull().sum() #统计年龄值缺失值个数
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count) #生成年龄分布的随机数
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list #年龄填补缺失值为年龄的正态分布值
    dataset['Age'] = dataset['Age'].astype(int) #将年龄变量转为int型
    dataset['Title'] = dataset['Name'].apply(get_title) #获取新变量 title
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare') #将不常见的title替换为Rare
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') #title替换
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int) #映射性别值为0-女，1-男
    # SexbyPclass
    dataset.loc[dataset['Sex']==0, 'SexByPclass'] = dataset.loc[dataset['Sex']==0, 'Pclass']
    dataset.loc[dataset['Sex']==1, 'SexByPclass'] = dataset.loc[dataset['Sex']==1, 'Pclass'] + 3
    dataset['SexByPclass'] = dataset['SexByPclass'].astype(int)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} #title的mapping list
    dataset['Title'] = dataset['Title'].map(title_mapping) #映射title值为1-5
    dataset['Title'] = dataset['Title'].fillna(0) #title填补缺失值为0
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) #映射Embarked值为0,1,2
    # 按不同范围重新映射Fare值,4分位数
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    # 按不同范围重新映射Age值，5分位数
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4    
    
    return dataset

# 分位数用法：pd.qcut(dataset['Fare'], 4)
#丢弃无用的描述变量
train = feature_reshape(train)
test = feature_reshape(test)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Surname', 'Cabin']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

'''
#相关系数图
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# 发现变量间的相关程度不高，适合模型的输入，只有Family_size和Parch两个变量间的相关系数达到0.78

# Pairplots
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
'''

# Ensembling & Stacking models
ntrain = train.shape[0] #891
ntest = test.shape[0] #418
SEED = 666 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# 新建一个Sklearn classifier类，将训练，预测等步骤合到一起
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
        
# Out-of-Fold Predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,)) # 1*891
    oof_test = np.zeros((ntest,))   # 1*418
    oof_test_skf = np.empty((NFOLDS, ntest)) # 5*418

    for i, (train_index, test_index) in enumerate(kf): # x_train: 891*11
        x_tr = x_train[train_index] # 712*10
        y_tr = y_train[train_index] # 712*1
        x_te = x_train[test_index]  # 179*10

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te) # 1*179 ===> will be 1*891 after 5 folds
        oof_test_skf[i, :] = clf.predict(x_test) # 1*418 ===> 5*418

    oof_test[:] = oof_test_skf.mean(axis=0) # 1*418 取5个模型的平均
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) # 819*1, 418*1

'''
Base First-Level Models:
    Random Forest classifier
    Extra Trees classifier
    AdaBoost classifer
    Gradient Boosting classifer
    Support Vector Machine
'''

#各个模型的参数设置
rf_params = {
    'n_jobs': -1,
    'n_estimators': 300,
    'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':300,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 300,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 300,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

# 特征重要性
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

# 将5个模型特征重要性放到一个dataframe里
cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
     'AdaBoost feature importances': ada_feature,
     'Gradient Boost feature importances': gb_feature
    })
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) #计算每个特征的重要性均值

'''
# Plotly Barplot of Average Feature Importances
y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')
'''

# Second-Level Predictions from the First-level Output
# First-level output as new features
base_predictions_train = pd.DataFrame({
     'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
     'GradientBoost': gb_oof_train.ravel()
    })


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

#Second level learning model via XGBoost
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)