from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers import Dense, Dropout, BatchNormalization, LSTM


def create_nn_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # No activation function for output layer in regression
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])  # Using Mean Squared Error for regression
    return model


def create_lstm_classification_model(input_shape, optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
