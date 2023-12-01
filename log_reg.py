import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import configparser

#Config parser
config = configparser.ConfigParser()
config.read('autoclassifier_config.properties')
ds_location = config.get('Section','ds_location')

ds=pd.read_csv(ds_location)
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix for Logistic Regression is:")
print(cm)
print("\nAccuracy Score for Logistic Regression is:")
print(accuracy_score(y_test,y_pred))

global lr_ac
lr_ac=accuracy_score(y_test,y_pred)

