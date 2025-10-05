#-------------------------------------------------------------------------
# AUTHOR: Jessie Mendoza
# FILENAME: naive_bayes.py
# SPECIFICATION: classify the instances of playing tennis using Naive Bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
outlook_map = {'Sunny': 0, 'Overcast': 1, 'Rain': 2}
temp_map = {'Hot': 0, 'Mild': 1, 'Cool': 2}
humid_map = {'High': 0, 'Normal': 1}
wind_map = {'Weak': 0, 'Strong': 1}

for row in dbTraining:
    X.append([
            outlook_map[row[1]],
            temp_map[row[2]],
            humid_map[row[3]],
            wind_map[row[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
class_map = {'No': 0, 'Yes': 1}
for row in dbTraining:
    Y.append(class_map[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print(f"{'Day':<8}{'Outlook':<12}{'Temperature':<15}{'Humidity':<10}{'Wind':<10}{'PlayTennis':<12}{'Confidence':<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
rev_class_map = {v: k for k, v in class_map.items()}
for row in dbTest:
    x_test = [[
            outlook_map[row[1]],
            temp_map[row[2]],
            humid_map[row[3]],
            wind_map[row[4]]
    ]]
    probs = clf.predict_proba(x_test)[0]
    class_predicted = probs.argmax()
    confidence = probs[class_predicted]

    if confidence >= 0.75:
        print(f"{row[0]:<8}{row[1]:<12}{row[2]:<15}{row[3]:<10}{row[4]:<10}{rev_class_map[class_predicted]:<12}{confidence:<10.2f}")








