import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #ml model with multiple decision trees
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head() #loading dataset

df.info() #memory usage, no. of rows and columns, data types, non-null
df.describe() #mean, standard dev, min, 25%...median
df.isnull().sum() #check for missing values 

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# convert values from strings to float
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
# null values are replaced with median of the column 
df.drop(['customerID'], axis=1, inplace=True)
#drop the column we don't need it

from sklearn.preprocessing import LabelEncoder

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#convert churn values to binary 
df = pd.get_dummies(df, drop_first=True)
#prevent redudant infromation by one-hot encoding remainig categorical columns

df['Churn'].value_counts().plot(kind='bar', title='Churn Class Balance');
# creates a bar chat of the churn column spread

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
#create a heat map to find highly correlated predictors and to guide feature selection for models
plt.show()

#split the data for traning and testing
X = df.drop('Churn', axis=1)
y = df['Churn'] #values we want to predict 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) #test_size reserves 20% of the data for testing

#using random forest classifier to predict churn 
model = RandomForestClassifier(n_estimators=100, random_state=42)#use 100 trees in the forest 
model.fit(X_train, y_train) 

#TP= 1 churn, FP=Incorrect churn predicted, FN= model missed a churn, TN= 0 churn 
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred)) #prints the FP, TP, FN, TN matrix
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1])) 

importances = pd.Series(model.feature_importances_, index=X.columns) 
#assigns a score for each feature, to show how important it was for decision trees 
importances.nlargest(10).plot(kind='barh')  #picks top 10 imp features 

import joblib

# Save model
joblib.dump(model, 'churn_model.pkl')
# Save column names
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')