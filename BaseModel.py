from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import colorama as colorama
import keras.backend
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from numpy import array
from colorama import Fore

total_accuracies = []
total_differences = []


def start_model(training_data_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    overall_data = array(pd.read_csv(training_data_path), dtype=float)
    train_data = overall_data
    # train_data = overall_data[:round(len(overall_data) / 2)]
    eval_data = overall_data[round(len(overall_data) / 2):len(overall_data)]

    train_features = train_data.copy()
    train_labels = train_features[:, 3]

    eval_features = eval_data.copy()
    eval_labels = eval_features[:, 3]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,)),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1),
    ])

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=len(train_data) * 1000,
        decay_rate=1,
        staircase=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    model.fit(train_features, train_labels, batch_size=round(len(train_data) / 16), epochs=10,
              steps_per_epoch=16,
              shuffle=True, verbose=0)

    average_accuracy = plot_predictions(model, eval_features, eval_labels)

    while average_accuracy < 95:
        keras.backend.clear_session()
        model.fit(train_features, train_labels, batch_size=round(len(train_data) / 16), epochs=10,
                  steps_per_epoch=16,
                  shuffle=True, verbose=0)
        average_accuracy = plot_predictions(model, eval_features, eval_labels)

    return model, eval_features, eval_labels


def plot_predictions(model, eval_features, eval_labels):
    predictions = model.predict(eval_features)

    average_accuracy = 0

    for i in range(len(eval_features)):
        difference = round(predictions[i][0] - eval_features[i][0], 5)
        percent_error = round((difference / eval_features[i][0]) * 100, 5)
        accuracy = round(100 - abs(percent_error), 5)
        average_accuracy += accuracy

    average_accuracy /= len(eval_features)
    print('Average accuracy of model is: ' + str(average_accuracy))

    plt.figure(figsize=(10, 10))
    plt.scatter(eval_labels, predictions, c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(predictions), max(eval_labels))
    p2 = min(min(predictions), min(eval_labels))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.savefig('Resources/fig.png')

    return average_accuracy


def predict_and_check(previous_prices_list, actual_price_str, actual_price_float, model):
    # predictions = model.predict_on_batch(array([[previous_prices_list]]))
    # print(predictions)
    prediction = round(model.predict(previous_prices_list)[-1:][0][0], 5)
    difference = round(prediction - actual_price_float, 5)
    percent_error = round((difference / actual_price_float) * 100, 5)
    accuracy = round(100 - abs(percent_error), 5)
    total_accuracies.append(accuracy)
    total_differences.append(difference)

    if accuracy >= 85:
        accuracy_color = Fore.GREEN
    elif accuracy >= 70:
        accuracy_color = Fore.YELLOW
    else:
        accuracy_color = Fore.RED

    average_accuracy = round(np.array(total_accuracies).mean(), 5)
    if average_accuracy >= 85:
        average_accuracy_color = Fore.GREEN
    elif average_accuracy >= 70:
        average_accuracy_color = Fore.YELLOW
    else:
        average_accuracy_color = Fore.RED

    print(f'{Fore.WHITE}Newest prediction was: {Fore.LIGHTBLUE_EX}{str(prediction)}')
    print(f'{Fore.WHITE}Actual price was: {Fore.GREEN}{actual_price_str}')
    print(f'{Fore.WHITE}Difference was: {Fore.LIGHTRED_EX}{difference}')
    print(f'{Fore.WHITE}Mean Difference is: {Fore.LIGHTRED_EX}{round(np.array(total_differences).mean(), 5)}')
    print(f'{Fore.WHITE}Accuracy was: {accuracy_color}{accuracy}')
    print(f'{Fore.WHITE}Mean Accuracy is: {average_accuracy_color}{average_accuracy}')
    print(f'{Fore.WHITE}+------------------------------------+')


def train_on_batched_live_data(live_data, open_price, number_to_batch, model):
    live_data_array = live_data[-number_to_batch:]
    live_max = np.amax(live_data_array)
    live_min = np.amin(live_data_array)
    live_data_array[:, 1] = open_price
    live_data_array[:, 2] = live_max
    live_data_array[:, 3] = live_min

    batched_labels = live_data_array[:, 0]
    model.fit(live_data_array, batched_labels, batch_size=1, epochs=2, shuffle=True, verbose=0)
