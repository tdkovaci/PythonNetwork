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
from sklearn.preprocessing import MinMaxScaler

total_accuracies = []
total_differences = []


def start_model(training_data_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    historical_data = np.array(pd.read_csv(training_data_path))

    response = requests.get('https://api.pro.coinbase.com/products/DOGE-USD/candles?granularity=60').json()
    overall_data = pd.DataFrame(response, columns=['Timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
    overall_data.drop('Timestamp', 1, inplace=True)
    overall_data = np.array(overall_data)

    x_train = []
    y_train = []
    for i in range(5, 300):
        x_train.append(overall_data[i - 5:i, 0])
        y_train.append(overall_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_hist = []
    y_hist = []
    for i in range(5, 300):
        x_hist.append(historical_data[i - 5:i, 0])
        y_hist.append(historical_data[i, 0])
    x_hist, y_hist = np.array(x_hist), np.array(y_hist)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_hist = np.reshape(x_hist, (x_hist.shape[0], x_hist.shape[1], 1))

    try:
        model = keras.models.load_model('Resources/doge_model')
    except OSError:

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        model.fit(x_train, y_train, epochs=100, batch_size=8)

        model.save('Resources/doge_model')

    return model, x_hist, y_hist


def plot_predictions(model, eval_features, eval_labels):
    predictions = model.predict(eval_features)
    average_accuracy = 0

    for i in range(len(eval_features)):
        difference = round((predictions[i][0] - eval_features[i][0])[0], 5)
        percent_error = round((difference / eval_features[i][0])[0] * 100, 5)
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


def predict_and_check(previous_prices_list, actual_price_float, model):
    # predictions = model.predict_on_batch(array([[previous_prices_list]]))
    # print(predictions)
    previous_prices_list = np.array(previous_prices_list)
    previous_prices_list = np.reshape(previous_prices_list, (previous_prices_list.shape[0], previous_prices_list.shape[1], 1))

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

    actual_price_str = str(actual_price_float)

    print(f'{Fore.WHITE}Newest prediction was: {Fore.LIGHTBLUE_EX}{str(prediction)}')
    print(f'{Fore.WHITE}Actual price was: {Fore.GREEN}{actual_price_str}')
    print(f'{Fore.WHITE}Difference was: {Fore.LIGHTRED_EX}{difference}')
    print(f'{Fore.WHITE}Mean Difference is: {Fore.LIGHTRED_EX}{round(np.array(total_differences).mean(), 5)}')
    print(f'{Fore.WHITE}Accuracy was: {accuracy_color}{accuracy}')
    print(f'{Fore.WHITE}Mean Accuracy is: {average_accuracy_color}{average_accuracy}')
    print(f'{Fore.WHITE}+------------------------------------+')


def train_on_batched_live_data(live_data, model):
    print(f'XXXXXXXXX Re-training on past 300 live data prices XXXXXXXXX')

    live_data = np.array(live_data)

    x_train = []
    y_train = []
    for i in range(5, 300):
        x_train.append(live_data[i - 5:i, 0])
        y_train.append(live_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model.fit(x_train, y_train, batch_size=8, epochs=100, shuffle=True, verbose=0)
    model.save('Resources/doge_model')
    print('XXXXXXXXX Finished training! XXXXXXXXX')
