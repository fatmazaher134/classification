import pandas as pd
from seaborn.categorical import countplot

#import the data set
employee=pd.read_csv('hr_employee_churn_data.csv')

# print(employee.head(5))

# null value 
employee.isna().any()

# drop null value
def impute_null_value(columns):
    satisfaction_level=columns[0]
    left=columns[1]
    if pd.isnull(satisfaction_level):
        if left==1:
            return employee[employee['left']==1]['satisfaction_level'].mean()
        elif left==0:
            return employee[employee['left']==0]['satisfaction_level'].mean()
    else:
        return satisfaction_level


employee['satisfaction_level']=employee[['satisfaction_level','left']].apply(impute_null_value,axis=1)
# employee.isna().any()



# change salary column
for i ,j in zip(employee['salary'], range(0,employee['salary'].count())):
    if i=='low':
        employee['salary'][j]=0
    elif i=='medium':
        employee['salary'][j]=1
    elif i=='high':
        employee['salary'][j]=2

    
# split the data into x and y
y=employee['left']
x=employee.drop(['empid','left'],axis=1)
# print(x.head(5))

# split the data into traning and test
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)

# create the model
from sklearn.linear_model import LogisticRegression 
# train and create prediction
model=LogisticRegression()
model.fit(x_train,y_train,sample_weight=None)
predict=model.predict(x_test)

# calculate performance
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))
