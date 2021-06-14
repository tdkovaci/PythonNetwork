from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


def split_data_set(input_file):
    for i in range(len(input_file)):
        if i == 0:
            split_file = open(str('Resources/DogeCoinTrainingSet.csv'), 'w+')
            split_file.writelines(input_file[i:i + (round(len(input_file) / 2))])
        if i == len(input_file) / 2:
            split_file = open(str('Resources/DogeCoinEvalSet.csv'), 'w+')
            split_file.write("Date,Closing Price,Open,High,Low\n")
            split_file.writelines(input_file[i:i + (round(len(input_file) / 2))])


def make_input_fn(data, target, epochs=5, shuffle=True, batch_size=8):
    def input_function():
        data_set = tf.data.Dataset.from_tensor_slices((dict(data), target))
        if shuffle:
            data_set = data_set.shuffle(1000)
        data_set = data_set.batch(batch_size).repeat(epochs)
        return data_set

    return input_function()


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
csv_file = open('Resources/DogeCoinInitialSet.csv', 'r').readlines()
split_data_set(csv_file)

train_data = pd.read_csv('Resources/DogeCoinTrainingSet.csv')
eval_data = pd.read_csv('Resources/DogeCoinEvalSet.csv')

train_features = train_data.copy()
train_labels = train_features.pop('Closing Price')

eval_features = eval_data.copy()
eval_labels = eval_features.pop('Closing Price')

train_features.pop("Date")

dataset = tf.data.Dataset.from_tensor_slices((train_features.values, train_labels.values))
train_dataset = dataset.shuffle(len(train_features)).batch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError()
)

print(model.fit(train_dataset, epochs=15))
