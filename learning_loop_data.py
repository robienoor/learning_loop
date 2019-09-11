from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\numerical_converted_data_drugslib.csv', delimiter=",")


X = data[0:2,:].T

Y_raw = data[2] / 10

print(Y_raw)

for idx, y in enumerate(Y_raw):
    if y < 0.6:
        Y_raw[idx] = 0
    else:
        Y_raw[idx] = 1

model = Sequential([
    Dense(50, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1, activation='sigmoid'),
])


model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X,
    Y_raw,
    epochs = 50,
)

predictions = model.predict(X)

np.savetxt('nn_predictions.csv', predictions,delimiter=",")
np.savetxt('ratings.csv', Y_raw, delimiter=",")


regressor = RandomForestRegressor(n_estimators=25, random_state=0)
regressor.fit(X, Y_raw)
y_pred = regressor.predict(X)

np.savetxt('forest.csv', y_pred)

print('forest: ', mean_squared_error(Y_raw, y_pred))
print('nn: ', mean_squared_error(Y_raw, predictions))


