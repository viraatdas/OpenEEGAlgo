#sklearn modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#pandas
import pandas as pd

#data visualization
import seaborn as sn
import matplotlib.pyplot as plt


#importing training data
dir = "/Users/owner/Documents/GitHub/OpenEEGAlgo/EEG Recordings/Songs/"
noah_dukas_1 = pd.read_csv(dir + "Dukas (The Sorceres Apprentice_Dukas)/Noah_Dukas/Noah_Dukas_take1.csv", header=None)
noah_esp3_1 = pd.read_csv(dir + "Esp3/Noah_Esp3/Noah_Esp3_take1.csv", header=None)
noah_opop_1 = pd.read_csv(dir + "OpOp/Noah_OpOp/Noah_OpOp_take1.csv", header=None)
noah_over_1 = pd.read_csv(dir + "Over (1812 Overture_Tchaikovsky)/Noah_Over/Noah_Over_take1.csv", header=None)
noah_se8_1 = pd.read_csv(dir + "Se8/Noah_Se8/Noah_Se8_take1.csv", header=None)

#importing test data
noah_dukas_2 = pd.read_csv(dir + "Dukas (The Sorceres Apprentice_Dukas)/Noah_Dukas/Noah_Dukas_take2.csv", header=None)
noah_esp3_2 = pd.read_csv(dir + "Esp3/Noah_Esp3/Noah_Esp3_take2.csv", header=None)
noah_opop_2 = pd.read_csv(dir + "OpOp/Noah_OpOp/Noah_OpOp_take2.csv", header=None)
noah_over_2 = pd.read_csv(dir + "Over (1812 Overture_Tchaikovsky)/Noah_Over/Noah_Over_take2.csv", header=None)
noah_se8_2 = pd.read_csv(dir + "Se8/Noah_Se8/Noah_Se8_take2.csv", header=None)

#inserting indicators as classifiers
noah_dukas_1.insert(16, 16, 0)
noah_esp3_1.insert(16, 16, 1)
noah_opop_1.insert(16, 16, 2)
noah_over_1.insert(16, 16, 3)
noah_se8_1.insert(16, 16, 4)

#inserting indicators for test data so we can test the classifier later on
noah_dukas_2.insert(16, 16, 0)
noah_esp3_2.insert(16, 16, 1)
noah_opop_2.insert(16, 16, 2)
noah_over_2.insert(16, 16, 3)
noah_se8_2.insert(16, 16, 4)

#concatenating dataset
trainData = pd.concat([noah_dukas_1, noah_esp3_1, noah_opop_1, noah_over_1, noah_se8_1]).values
testData = pd.concat([noah_dukas_2, noah_esp3_2, noah_opop_2, noah_over_2, noah_se8_2]).values

#preprocessing train data
X_train = trainData[:, :-1] #data
Y_train = trainData[:, 16]  #labels

#prepreocessing test data
X_test = trainData[:, :-1]  #data
Y_test = trainData[:, 16]   #labels

#splitting the data into 80% train data and 20% test data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#feature scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#training
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)

#predictions
y_pred = classifier.predict(X_test)

#visualize and print data
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
sn.heatmap(confusion_matrix(Y_test, y_pred))
plt.show()

