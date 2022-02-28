#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('dataset.csv')
#print(type(loan_dataset))
#print(loan_dataset.describe())

# dropping the missing values
loan_dataset = loan_dataset.dropna()

# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1}, 'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

#Seprating Training and Testing Data
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

#TRAING SVM MODEL
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ', training_data_accuray)

X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ', test_data_accuray)


#PREDICTIVE SYSTEM

input_data=[]
val1=input("Enter Gender : ")
if(val1.lower()=='male'):
  input_data.append(1)
else:
  input_data.append(0)

val2=input("Enter Marital Status : ")
if(val2.lower()=='married'):
  input_data.append(1)
else:
  input_data.append(0)

val3=int(input("Enter No. of Dependants : "))
input_data.append(val3)

val4=input("Enter Educational Qualification(Graduate/Not Graduate) : ")
if val4.lower()=="graduate":
  input_data.append(1)
else:
  input_data.append(0)

val5=input("Enter Self Employed Status(Y/N) : ")
if val5.lower()=='y':
  input_data.append(1)
else:
  input_data.append(0)

val6=int(input("Enter Applicant Income : "))
input_data.append(val6)

val7=int(input("Enter Co-Applicant Income : "))
input_data.append(val7)

val8=int(input("Enter Loan Amount : "))
input_data.append(val8)

val9=int(input("Enter Loan Amount Term : "))
input_data.append(val9)

val10=int(input("Enter Credit History : "))
input_data.append(val10)

val11=input("Enter Property Area : ")
if val11.lower()=='rural':
  input_data.append(0)
elif val11.lower()=='semiurban':
  input_data.append(1)
else:
  input_data.append(2)

print(input_data)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='Y'):
  print('Loan is Approved')
else:
  print('Loan is Rejected')