from pathlib import Path
import datetime as dt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
import os
import re

data_dir = Path.cwd() / 'data' / 'ml'
today = dt.date.today().strftime("%m-%d-%Y")
# tickers = stocks.gather_wsb_tickers(data_dir, today)

# model = keras.models.load_model('test')

files = []

for file in os.listdir(data_dir):

    match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

    if match and 'normalized' in file:
        files.append(file)
        # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

files = sorted(files, key=lambda x: dt.datetime.strptime(
    re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

x = len(files)
bool = True
for file in files:
    path = data_dir / file
    date_str = re.search(r'\d\d-\d\d-\d\d\d\d', file)[0]
    data = pd.read_csv(path)
    data.rename(columns={'Unnamed: 0': 'ticker'}, inplace=True)
    features = data.copy()
    labels = features.pop('Y')

    # some preprocessing
    inputs = {}

    for name, column in features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}
    all_numeric_inputs = layers.Concatenate()(list(numeric_inputs.values()))
    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = preprocessing.StringLookup(vocabulary=np.unique(features[name]))
        one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    # tf.keras.utils.plot_model(model=data_preprocessing, rankdir="LR", dpi=72, to_file="test.png", show_shapes=True)

    features_dict = {name: np.array(value)
                             for name, value in features.items()}
    features_dict = {name: values[:1] for name, values in features_dict.items()}
    data_preprocessing(features_dict)

    def trading_model(preprocessing_head, inputs):
        body = tf.keras.Sequential([
            layers.Dense(64),
            layers.Dense(1)
        ])

        preprocessed_inputs = preprocessing_head(inputs)
        result = body(preprocessed_inputs)
        model = tf.keras.Model(inputs, result)

        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.optimizers.Adam())
        return model


    trading_model = trading_model(data_preprocessing, inputs)
    trading_model.fit(x=features_dict, y=labels, epochs=10)

    # titanic_model.save('test')
    # reloaded = tf.keras.models.load_model('test')

    if bool:
        y = len(features.index)
        z = len(features.columns)
        dataset = np.zeros((x,y,z))
        bool = False

    features = np.array(features)

print()
# data =  pd.read_csv(data_dir / ('normalized_data_' + today + '.csv'), index_col=0)
# features = data.copy()
# labels = features.pop('Y')
#
# features = np.array(features)
#
# model = tf.keras.Sequential([
#   layers.Dense(64),
#   layers.Dense(1)
# ])
#
# model.compile(loss = tf.losses.MeanSquaredError(),
#                       optimizer = tf.optimizers.Adam())
#
#
# model.fit(features, labels, epochs=10)
# # model.save('test')
#
# test =  pd.read_csv(data_dir / ('normalized_data_' + '04-10-2021' + '.csv'), index_col=0)
# test = np.array(test)
# predictions = model.predict(features)
#
# print(predictions)

