from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from keras.utils import plot_model

X = np.array([[1,0,0],
                [1,0,1],
                [1,1,1]])

Y = np.array([[1,0,0],
                [1,0,1],
                [1,1,1]])


model = Sequential([
    Dense(64, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='sigmoid'),
])

model.compile(
    optimizer = 'adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(
    X,
    Y,
    epochs = 50,
)

plot_model(model, to_file='model.png')

print(model.predict(X))