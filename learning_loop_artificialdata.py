import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import random

X = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\artificial_input.txt', delimiter=" ")
Y = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\artificial_labels.txt', delimiter=" ")
Y_args = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\arg_graph_labels.txt', delimiter=" ")

test = random.sample(range(1, len(Y_args)), 40)

# c = list(zip(X, Y_args))

# random.shuffle(c)

# X, Y_args = zip(*c)


model = Sequential([
    Dense(100, activation='relu'),
    Dense(200, activation='relu'),
    Dense(3, activation='sigmoid'),
])

model.compile(
    optimizer = 'adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(
    X,
    Y_args,
    epochs = 50,
)


predictions = model.predict(X)

np.savetxt('nn_predictions.csv', predictions,delimiter=",")
np.savetxt('ratings.csv', Y, delimiter=",")