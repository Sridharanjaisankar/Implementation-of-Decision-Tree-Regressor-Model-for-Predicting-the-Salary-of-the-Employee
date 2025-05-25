# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preprocessing: 1.1 Load the Dataset: Load the dataset from a CSV file (Salary.csv). 1.2 Check for Missing Data: Check if there are any missing values in the dataset. 1.3 Encode Categorical Data: If there is any categorical data (like the "Position" column), apply label encoding to convert it into numeric values.

Feature and Target Variables: 2.1 Define Features (X): Extract the feature columns ("Position" and "Level"). 2.2 Define Target (Y): Extract the target column ("Salary").

Split the Data: 3.1 Split the data into training and testing sets. This will allow us to train the model on one subset (training) and test its performance on another subset (testing). 3.2 Set aside 20% of the data for testing (test_size=0.2) and use the remaining 80% for training.

Model Training: 4.1 Initialize Decision Tree Regressor: Create an instance of the Decision Tree Regressor model. 4.2 Fit the Model: Train the model using the training data (X_train, Y_train).

Model Evaluation: 5.1 Predict on Test Set: Use the trained model to predict the salary values on the test set (X_test). 5.2 Calculate Mean Squared Error (MSE): Evaluate the model's performance using Mean Squared Error. This helps to understand how much the model's predictions differ from the actual values. 5.3 Calculate R² Score: Evaluate the model’s goodness of fit using the R² score. It indicates how well the model explains the variance in the target variable.

Prediction for New Data: 6.1 Predict for New Inputs: Use the model to make predictions for new inputs (e.g., Position=5, Level=6).

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SRIDHARAN J
RegisterNumber:  212222040158
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

DATA HEAD:

![Screenshot 2025-04-30 141828](https://github.com/user-attachments/assets/fe502337-c53b-474f-9888-8e76909827ea)


DATA INFO:

![Screenshot 2025-04-30 141835](https://github.com/user-attachments/assets/9056cb70-a0a9-420f-9680-90885a347ddd)


ISNULL() AND SUM():

![Screenshot 2025-04-30 141840](https://github.com/user-attachments/assets/7d57ef9f-0d5c-441a-b3c3-fe9ae25bedfa)

DATA HEAD FOR SALARY:

![Screenshot 2025-04-30 141846](https://github.com/user-attachments/assets/b5cbab73-9fad-4613-93c3-baafc2254d2e)

MEAN SQUARED ERROR:

![Screenshot 2025-04-30 141859](https://github.com/user-attachments/assets/e109e8f3-5cf7-40ef-9220-3f9015c41684)


R2 VALUE:

![Screenshot 2025-04-30 141906](https://github.com/user-attachments/assets/fcc3060b-89bf-4ec9-be43-ab84a12fadad)


DATA PREDICTION:

![Screenshot 2025-04-30 141921](https://github.com/user-attachments/assets/cfd95da9-21d5-4020-b95d-61097509913c)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
