import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import random

# Load Data
X = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\artificial_input.txt', delimiter=" ")
Y = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\artificial_labels.txt', delimiter=" ")
Y_args = np.loadtxt(r'C:\Users\Robie\Documents\Programming Learning\Python\learning_loop\dataset\drugLib_raw\arg_graph_labels.txt', delimiter=" ")
indices = range(len(X))
# Break data into train and test
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,Y_args,indices, test_size=0.05)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Construct model
model = Sequential([
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
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


predictions = model.predict(X_test)

np.savetxt('nn_predictions.csv', predictions,delimiter=",")
np.savetxt('nn_test_indices.csv', indices_test, delimiter=",") 