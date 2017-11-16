import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data=pd.read_csv('train.csv')
data=data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
y=data['Survived'].values
data=data.drop('Survived', axis=1)
data['Age']=data['Age'].fillna(data['Age'].median())
data['Age']=pd.to_numeric(data['Age'], errors='coerce')
d = {'male': '0',
     'female': '1',
    }
data.Sex=data.Sex.map(d)
data['Sex']=pd.to_numeric(data['Sex'], errors='coerce')
#print(data)

test_data=pd.read_csv('test.csv')
test_data=test_data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test_data['Age']=test_data['Age'].fillna(test_data['Age'].median())
test_data['Age']=pd.to_numeric(test_data['Age'], errors='coerce')
d = {'male': '0',
     'female': '1',
    }
test_data.Sex=test_data.Sex.map(d)
test_data['Sex']=pd.to_numeric(test_data['Sex'], errors='coerce')
#print(test_data)

#print(data.shape)
#print(test_data.shape)

#data_train, data_test, y_train,y_test = train_test_split(data,y,test_size=0.3,random_state=21,stratify=y)
knn= GaussianNB()
knn.fit(data, y)
prediction=knn.predict(test_data)
print(knn.score(test_data,prediction))
final_df=pd.DataFrame([test_data['PassengerId'],prediction],index=['PassengerId', 'Survived']).T
print(final_df)
final_df.to_csv("C:/Users/HP/Documents/DataScience/Kaggle/Titanic/final_df.csv")