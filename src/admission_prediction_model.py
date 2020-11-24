import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('input/Admission_Predict.csv')
del df['Serial No.']
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Chance of Admit '],axis = 1), df['Chance of Admit '],random_state = 42,test_size = 0.2, shuffle = True)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

models = [
    ['DecisionTree' ,DecisionTreeRegressor()], 
    ['LinearRegression', LinearRegression()],
    ['RandomForest', RandomForestRegressor()],
    ['KNeighbour' ,KNeighborsRegressor(n_neighbors= 2)],
    ['Adaboost', AdaBoostRegressor()],
    ['GradietBoost', GradientBoostingRegressor()],
    ['XGBRegressor' ,XGBRegressor()],
    ['CatBoost', CatBoostRegressor(logging_level = 'Silent')],
    ['Lasso' ,Lasso()],
    ['Ridge' ,Ridge()],
    ['BayesianRidge', BayesianRidge()],
    ['ElasticNet', ElasticNet()],
    ['HuberRegressor', HuberRegressor()]
]

print('results are ................')

for name,model in models :
    model = model
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print(name, np.sqrt(mean_squared_error(y_test,pred)))
    pickle.dump(model, open('input/models/{}.pkl'.format(name),'wb'))


