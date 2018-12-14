from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#importing all the files
dir = "/Users/owner/Documents/GitHub/OpenEEGAlgo/EEG Recordings/Songs/"
noah_dukas_1 = pd.read_csv(dir + "Dukas (The Sorceres Apprentice_Dukas)/Noah_Dukas/Noah_Dukas_take1.csv", header=None)
noah_esp3_1 = pd.read_csv(dir + "Esp3/Noah_Esp3/Noah_Esp3_take1.csv", header=None)
noah_opop_1 = pd.read_csv(dir + "OpOp/Noah_OpOp/Noah_OpOp_take1.csv", header=None)
noah_over_1 = pd.read_csv(dir + "Over (1812 Overture_Tchaikovsky)/Noah_Over/Noah_Over_take1.csv", header=None)
noah_se8_1 = pd.read_csv(dir + "Se8/Noah_Se8/Noah_Se8_take1.csv", header=None)

#inserting indicators as classifiers
noah_dukas_1.insert(16, 16, 0)
noah_esp3_1.insert(16, 16, 1)
noah_opop_1.insert(16, 16, 2)
noah_over_1.insert(16, 16, 3)
noah_se8_1.insert(16, 16, 4)

#preprocessing
dataset = pd.concat([noah_dukas_1, noah_esp3_1, noah_opop_1, noah_over_1, noah_se8_1]).values
X = dataset[:, :-1] #data
Y = dataset[:, 16] #labels

#splitting the data into 80% train data and 20% test data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

