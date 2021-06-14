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
            split_file.write("Currency,Date,Closing Price,Open,High,Low\n")
            split_file.writelines(input_file[i:i + (round(len(input_file) / 2))])


def make_input_fn(data, target, epochs=4, shuffle=True, batch_size=100):
    def input_function():
        data_set = tf.data.Dataset.from_tensor_slices((dict(data), target))
        if shuffle:
            data_set = data_set.shuffle(2000)
        data_set = data_set.batch(batch_size).repeat(epochs)
        return data_set

    return input_function()


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
csv_file = open('Resources/DogeCoinInitialSet.csv', 'r').readlines()
split_data_set(csv_file)

training_data = pd.read_csv('Resources/DogeCoinTrainingSet.csv')
eval_data = pd.read_csv('Resources/DogeCoinEvalSet.csv')

training_target_col = training_data.pop('Closing Price')
evaluation_target_col = eval_data.pop('Closing Price')
# print(evaluation_target_col)

category_cols = ['Currency', 'Date']
numeric_cols = ['Open', 'High', 'Low']

feature_cols = []
for feature_name in category_cols:
    vocab = training_data[feature_name].unique()
    feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in numeric_cols:
    feature_cols.append(tf.feature_column.numeric_column(feature_name))

training_function = lambda: make_input_fn(training_data, training_target_col)
evaluation_function = lambda: make_input_fn(eval_data, evaluation_target_col, 1, False)

linear_estimator = tf.estimator.LinearClassifier(feature_cols)
linear_estimator.train(training_function)
result = linear_estimator.evaluate(evaluation_function)

print(result['accuracy'])
