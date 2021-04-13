from pathlib import Path
import datetime as dt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data_dir = Path.cwd() / 'data' / 'ml'
today = dt.date.today().strftime("%m-%d-%Y")
# tickers = stocks.gather_wsb_tickers(data_dir, today)

data =  pd.read_csv(data_dir / ('normalized_data_' + today + '.csv'), index_col=0)
features = data.copy()
labels = features.pop('Y')

features = np.array(features)

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())


model.fit(features, labels, epochs=10)
# model.save('test')

test =  pd.read_csv(data_dir / ('normalized_data_' + '04-10-2021' + '.csv'), index_col=0)
test = np.array(test)
predictions = model.predict(features)

print(predictions)