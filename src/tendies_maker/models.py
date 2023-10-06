from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers import Dense, Dropout, BatchNormalization


def create_nn_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # No activation function for output layer in regression
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])  # Using Mean Squared Error for regression
    return model
