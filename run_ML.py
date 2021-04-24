from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import callbacks
import tensorflow as tf
import os
import re


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    try:
        labels = dataframe.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    except:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


# Configuration variables
data_dir = Path.cwd() / 'data' / 'ml'
today = dt.date.today().strftime("%m-%d-%Y")
# Set to 'train' or 'predict', see README.md for detailed instructions
runtype = 'train'
# Set to 'train' or 'load', see README.md for detailed instructions
model_source = 'train'


files = []

for file in os.listdir(data_dir):

    match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

    if match and 'normalized' not in file and 'data' in file:
        files.append(file)
        # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

files = sorted(files, key=lambda x: dt.datetime.strptime(
    re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

x = len(files)
dataframes = []
test = pd.DataFrame()
i = 0
for file in files:
    path = data_dir / file
    date_str = re.search(r'\d\d-\d\d-\d\d\d\d', file)[0]
    dataframe = pd.read_csv(path)
    dataframe = dataframe.fillna(0)
    dataframe.rename(columns={'Unnamed: 0': 'ticker'}, inplace=True)
    dataframe['date'] = date_str
    if runtype == 'predict':
        if i == 0:
            # This makes the most recent one test data. Comment if shuffling from one massive df
            test = dataframe
        else:
            dataframes.append(dataframe)
    elif runtype == 'train':

        dataframes.append(dataframe)
    else:
        print("Not a valid option entered for 'type'. See README")
        quit()

    i += 1

dataframe = pd.concat(dataframes, ignore_index=True)
dataframe['target'] = np.where(dataframe['Y'] == 1, 1, 0)
dataframe = dataframe.drop(columns=['Y'])
dates = dataframe.pop('date').to_frame()


if runtype == 'predict':
    test_dates = test.pop('date').to_frame()
    train, val = train_test_split(dataframe, test_size=0.2)
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    print(len(train), 'train examples')

elif runtype == 'train':
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    print(len(train), 'train examples')


# Check if any nan values
# print(pd.isnull(dataframe).sum(),pd.isnull(train).sum(),pd.isnull(val).sum(),pd.isnull(test).sum())

if model_source == 'train':
    batch_size = 16
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    all_inputs = []
    encoded_features = []

    # Numeric features.
    for header in ['rank', 'mentions', 'change', 'days_since_max_rise', 'days_since_last_rise',
                   'rsi', 'bb_val', 'macd', 'ichimoku', 'news_sentiment', 'DTE', 'max_intraday_change_1mo',
                   'put_call_ratio', 'put_call_value_ratio', 'sentiment', 'short_interest']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Categorical features encoded as string.
    categorical_cols = ['ticker']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="sigmoid")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001,)

    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(train_ds, epochs=100, validation_data=val_ds,callbacks=[reduce_lr, model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    model.save('model')

elif model_source == 'load':
    model = tf.keras.models.load_model('model')

# Get weights if needded
# for layer in model.layers:
#     print(layer.name)
#     print(layer.get_weights())
#     print()

# Accuracy, swap val_ds with test_ds depending on what data is tested
if runtype=='train':
    loss, accuracy = model.evaluate(val_ds)
elif runtype=='predict':
    loss, accuracy = model.evaluate(test_ds)

print("Accuracy", accuracy)
test_data_dict = test.to_dict(orient='records')
sample = test_data_dict[0]
test_input = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(x=test_ds)

# prob = np.argmax(predictions, axis=1)
prob = tf.nn.sigmoid(predictions)
# prob = ((predictions > 0.5)+0).ravel()
# prob = tf.nn.softplus(predictions)
prob = tf.keras.backend.get_value(prob % (100 * prob))

# Save to .csv
new_df = test[['ticker']]
# Make this 0 if using actual live test data.
new_df['pred'] = np.array(prob)
if runtype == 'train':
    new_df['actual'] = test[['target']]
    new_df = new_df.join(dates, how='left')
elif runtype == 'predict':
    new_df['actual'] = 0
    new_df = new_df.join(test_dates, how='left')


new_df = new_df.join(dates, how='left')
new_df.to_csv(data_dir / 'predictions.csv')
