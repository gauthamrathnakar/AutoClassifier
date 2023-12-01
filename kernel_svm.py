import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
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

classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

print("Confusion Matrix for SVM Non-Linear is:")
print(confusion_matrix(y_test,y_pred))
print("\nAccuracy score for SVM Non_linear is:")
print(accuracy_score(y_test,y_pred))

global svc_nl_ac
svc_nl_ac = accuracy_score(y_test,y_pred)