import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import configparser

#Config parser
config = configparser.ConfigParser()
config.read('autoclassifier_config.properties')
ds_location = config.get('Section','ds_location')

ds=pd.read_csv(ds_location)
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

sd=StandardScaler()
X_train=sd.fit_transform(X_train)
X_test=sd.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print("Confusion Matrix for KNN is:")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("\nAccuracy Score for KNN is:")
print(accuracy_score(y_test,y_pred))

global knn_ac
knn_ac=accuracy_score(y_test,y_pred)