#Importing Dependies
.  import numpy as np
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

#Data Standardization

.  scaler = StandardScaler()

.  scaler.fit(X_train)

.  X_train = scaler.transform(X_train)
   X_test = scaler.transform(X_test)

.  print(X_train)

#Model Training
#Support Vector Machine Model

.  model = svm.SVC(kernel='linear')

# training the SVM model with training data
.  model.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

# accuracy score on training data
.  X_train_prediction = model.predict(X_train)
   training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

.  print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
.  X_test_prediction = model.predict(X_test)
   test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

.  print ('Accuracy on Test data :' , test_data_accuracy)

#Building a Predictive System

. input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
  input_data_as_numpy_array = np.asarray(input_data)    # changing input data to a numpy array
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)     # reshape the numpy array
  std_data = scaler.transform(input_data_reshaped)    # standardize the data
  prediction = model.predict(std_data)
  print(prediction)
  if (prediction[0] == 0):
    print("The Person does not have Parkinsons Disease")
  else:
    print("The Person has Parkinsons")

