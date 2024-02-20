#Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler as ms
#Loading the dataset
df = pd.read_csv("/content/churn_df.csv")
df.head()
#Check for data balance
df["Exited"].value_counts()
#Drop unnecessary columns tenure and age
df = df.drop(columns = ["Tenure","Age"], axis = 1)
df.columns
#Split the dataset into target and feature
x = df.drop("Exited", axis = 1)
y=df["Exited"]
xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size = 0.33,random_state = 42,stratify = y)
#Normalising the daatset
m = ms()
m.fit(xtrain)
xtrain = m.transform(xtrain)
xtest = m.transform(xtest)
#Model creation
g = GaussianNB()
g.fit(xtrain,ytrain)
ypred = g.predict(xtest)
print(classification_report(ytest,ypred))
print(classification_report(ytest,ypred))