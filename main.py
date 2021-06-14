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

category_cols = ['Date']
numeric_cols = ['Open', 'High', 'Low']

feature_cols = []
for feature_name in category_cols:
    vocab = train_data[feature_name].unique()
    feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in numeric_cols:
    feature_cols.append(tf.feature_column.numeric_column(feature_name))

training_function = lambda: make_input_fn(train_data, train_labels)
evaluation_function = lambda: make_input_fn(eval_data, eval_labels, 1, False)

linear_estimator = tf.estimator.LinearClassifier(feature_cols)
linear_estimator.train(training_function)
result = linear_estimator.evaluate(evaluation_function)

print(result['average_loss'])
