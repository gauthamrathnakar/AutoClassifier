import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import configparser

#Config parser
config = configparser.ConfigParser()
config.read('autoclassifier_config.properties')
ds_location = config.get('Section','ds_location')


ds=pd.read_csv(ds_location)
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

ds=DecisionTreeClassifier(criterion='entropy',random_state=0)
ds.fit(X_train,y_train)

y_pred=ds.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix for Decision Tree is:")
print(cm)

print("\nAccuracy Score for Decision Tree is:")
global dt_ac
dt_ac = accuracy_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))