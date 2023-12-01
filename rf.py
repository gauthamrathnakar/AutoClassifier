import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import configparser

#Config parser
config = configparser.ConfigParser()
config.read('autoclassifier_config.properties')
ds_location = config.get('Section','ds_location')

ds=pd.read_csv(ds_location)
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

rc=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
rc.fit(X_train,y_train)

y_pred=rc.predict(X_test)
print("Confusion matrix for Random Forest is:")
print(confusion_matrix(y_test,y_pred))
print("\nAccuracy Score for Random Forest is:")
print(accuracy_score(y_test,y_pred))

global rf_ac
rf_ac = accuracy_score(y_test,y_pred)
       
