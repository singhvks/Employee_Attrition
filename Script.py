import pandas as pd

df = pd.read_csv("CSV location")

# Finding the missing values
#validate = df.isna() # No missing data found
Y = df['Attrition']
df = df.drop(columns = ['Attrition','EmployeeNumber','EmployeeCount','StandardHours'])

# Feature Encoding
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
df['Gender']= le.fit_transform(df['Gender']) 
df['Over18']= le.fit_transform(df['Over18']) 
df['OverTime']= le.fit_transform(df['OverTime']) 
df['BusinessTravel']= le.fit_transform(df['BusinessTravel']) 
df['Department']= le.fit_transform(df['Department']) 
df['EducationField']= le.fit_transform(df['EducationField']) 
df['JobRole']= le.fit_transform(df['JobRole']) 
df['MaritalStatus']= le.fit_transform(df['MaritalStatus']) 

# Target Variable transformed to binary
Y = le.fit_transform(Y)

# Extracting Nominal Features

NominalData = df[['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(NominalData)
data = enc.transform(NominalData).toarray()

# Gluing these features back to the original dataframe
df = df.drop(columns = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'])
data1 = pd.DataFrame(data)
df = pd.concat([df,data1], axis=1, join='outer')

# Testing and Training data Split
from sklearn.model_selection import train_test_split
Train_x, Test_x, Train_y, Test_y = train_test_split(df, Y, test_size=0.33, random_state=40)

#Model training
from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(Train_x,Train_y)

predict = reg.predict(Test_x)

from sklearn.metrics import accuracy_score
accuracy_score(Test_y, predict)

from sklearn.metrics import confusion_matrix
confu_mat = confusion_matrix(Test_y, predict)
