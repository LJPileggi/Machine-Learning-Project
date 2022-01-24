import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dataframe = pd.read_csv("../data/mlcup_internaltrain.csv", header=None)
dataset = dataframe.values

X = dataset[:, 0:10]
Y = dataset[:, 10:12]

def build_model():
    model = Sequential()
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))

    opt = SGD(learning_rate=0.001, momentum=0.5)
    
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

model = build_model()

model.fit(x=X, y=Y, batch_size=64, epochs=100, validation_split=0.2)

#estimator = KerasRegressor(build_fn=build_model, epochs=100, batch_size=64)

#kfold = KFold(n_splits=5)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
