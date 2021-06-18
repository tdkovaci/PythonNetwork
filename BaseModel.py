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


def determine_price_data(new_price, open_price, high_price, low_price, length_of_previous_prices):
    temp_open_price, temp_high_price, temp_low_price = 0, 0, 0

    if open_price == 0 or length_of_previous_prices % 10 == 0:
        temp_open_price = new_price

    if new_price > high_price:
        temp_high_price = new_price
    else:
        temp_high_price = high_price

    if new_price < low_price or low_price == 0:
        temp_low_price = new_price
    else:
        temp_low_price = low_price

    return new_price, temp_open_price, temp_high_price, temp_low_price


total_accuracies = []
total_differences = []


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


def train_on_batched_live_data(batched_data, model):
    batched_labels = batched_data[:, 0]
    model.fit(batched_data, batched_labels, batch_size=1, epochs=2, shuffle=True, verbose=0)


def batch_live_data(live_data, open_price, number_to_batch):
    live_data_array = live_data[-number_to_batch:]
    live_max = np.amax(live_data_array)
    live_min = np.amin(live_data_array)
    live_data_array[:, 1] = open_price
    live_data_array[:, 2] = live_max
    live_data_array[:, 3] = live_min

    return live_data_array


def get_new_price(url):
    response = requests.get(url).json()
    new_price_str = response['data']['prices'][0]['price']
    new_price_float = float(new_price_str)
    return new_price_str, new_price_float


def collect_live_data(model, url):
    number_to_batch = 10
    loop_iterator = 0
    previous_prices = []
    open_price = 0
    high_price = 0
    low_price = 0
    detected_change = True
    while True:

        if loop_iterator % 5 == 0:
            new_price_str, new_price_float = get_new_price(url)
            new_price_float, open_price, high_price, low_price = determine_price_data(new_price_float, open_price,
                                                                                      high_price, low_price,
                                                                                      len(previous_prices))
            if len(previous_prices) == 0:
                previous_prices = [[new_price_float, open_price, high_price, low_price]]
                predict_and_check(previous_prices, new_price_str, new_price_float, model)
                detected_change = False
            elif new_price_float != (previous_prices[-1:][0][0]):
                detected_change = True

            if detected_change:
                previous_prices = np.vstack(
                    [previous_prices, [new_price_float, open_price, high_price, low_price]])
                predict_and_check(previous_prices, new_price_str, new_price_float, model)
                detected_change = False
            else:
                detected_change = False

        # if len(previous_prices) % number_to_batch == 0:
        #     batched_data = batch_live_data(previous_prices, open_price, number_to_batch)
        #     train_on_batched_live_data(batched_data, model)
        #     print('XXXXXXXXX Re-training on past ' + str(number_to_batch) + ' live data prices XXXXXXXXX')
        #     time.sleep(3)
        #     previous_prices = []

        time.sleep(1)
        loop_iterator += 1


def main():
    model, eval_features, eval_labels = start_model('Resources/DOGE-USD.csv')

    collect_live_data(model, 'https://chain.so/api/v2/get_price/DOGE/USD')


if __name__ == "__main__":
    main()
