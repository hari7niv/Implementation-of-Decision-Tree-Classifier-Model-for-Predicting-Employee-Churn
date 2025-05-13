# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Read CSV data, handle nulls, encode categorical features like "salary".
2. Feature-target split: Select relevant features for x and set y as the "left" column.
3. Train-test split & modeling: Split the data and train a DecisionTreeClassifier using the "entropy" criterion.
4. Evaluate & predict: Measure accuracy on the test set and make predictions on new data.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sujith A
RegisterNumber:  212224230278
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![Screenshot 2025-05-12 104814](https://github.com/user-attachments/assets/27e6962a-759c-46a3-90f7-3bb5ad3a9c4d)
![Screenshot 2025-05-12 104826](https://github.com/user-attachments/assets/d7496228-a452-4796-a3d7-0101c48e1fe0)
![Screenshot 2025-05-12 105128](https://github.com/user-attachments/assets/562bd9de-aa6e-4348-8d12-a8868f0805e9)
![Screenshot 2025-05-12 105136](https://github.com/user-attachments/assets/98f91e17-fb7e-43b4-9ccb-3cdaba60dca8)
![Screenshot 2025-05-12 105147](https://github.com/user-attachments/assets/2577c4cf-eff0-4f3a-841b-84d99de81043)
![Screenshot 2025-05-12 105158](https://github.com/user-attachments/assets/cfd574eb-5d68-4168-a669-b4488d1814e2)
![Screenshot 2025-05-12 105207](https://github.com/user-attachments/assets/0d601580-8319-41a3-979b-dcfdff0cae66)
![Screenshot 2025-05-12 105214](https://github.com/user-attachments/assets/472c9267-c372-4974-918a-0db3b277a8d3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
