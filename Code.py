#Importing Dependies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis

# loading the csv data to a Pandas DataFrame
.  parkinsons_data = pd.read_csv('/content/parkinsons.csv')


# printing the first 5 rows of the dataset
.  parkinsons_data.head()

# printing last 5 rows of the dataset
.  parkinsons_data.tail()

#number of rows and columns in this dataset
.  parkinsons_data.shape

#getting some info 
.  parkinsons_data.info()

#checking for missing values
.  parkinsons_data.isnull().sum()

# getting the  statistical measures of the data
.  parkinsons_data.describe()


#checking the distribution of  target variable
.  parkinsons_data['status'].value_counts()

#1 --> Parkinson's Positive
#0 --> Healthy

# grouping the data bas3ed on the target variable
.  parkinsons_data.groupby('status').mean()


#Data Pre-Processing

#Splitting the features and target

.  X = parkinsons_data.drop(columns=['name','status'], axis=1)
   Y = parkinsons_data['status']

.  print(X)

.  print(Y)


#Splliting the Data into Training Data and Test Data
.  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,  random_state=2)

.  print(X.shape , X_train.shape,X_test.shape)

#Model Training
#Logistic Regression

.  model = LogisticRegression()

# training the logistic Regression model with Training data
.  model.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

# accuracy on training data
.  X_train_prediction = model.predict(X_train)
   training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

.  print ('Accuracy on training data :' , training_data_accuracy)

.  X_test_prediction = model.predict(X_test)
   test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

.  print ('Accuracy on Test data :' , test_data_accuracy)

#Building a Predictive System

.  input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
   input_data_as_numpy_array = np.array(input_data)      #change the input data to a numpy array
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)   #reshape the numpy array as we are predicting for only one instance
   prediction = model.predict(input_data_reshaped)
   print(prediction)
   if(prediction[0] == 0):
      print('The person does not have a Heart Disease')
  else:
      print('The persom has heart Disease')
     

# testing the models with 2 example(above is for defective heart case and the below one ia for healthy heart case


.  input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
   input_data_as_numpy_array = np.array(input_data)      #change the input data to a numpy array
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)    #reshape the numpy array as we are predicting for only one instance
   prediction = model.predict(input_data_reshaped)
   print(prediction)
  if(prediction[0] == 0):
    print('The person does not have a Heart Disease')
  else:
    print('The persom has heart Disease')



                 
