from __future__ import absolute_import, division, print_function, unicode_literals

from colorama import Fore
import keras.backend
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

total_accuracies = []
total_differences = []


def start_model(model_path, api_url, number_to_batch):
    response = requests.get(api_url).json()
    overall_data = pd.DataFrame(response, columns=['Timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
    overall_data.drop('Timestamp', 1, inplace=True)
    overall_data = np.array(overall_data)

    x_train = []
    y_train = []
    for i in range(5, 300):
        x_train.append(overall_data[i - 5:i, 0])
        y_train.append(overall_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    try:
        model = keras.models.load_model(model_path)
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

        model.fit(x_train, y_train, epochs=100, batch_size=number_to_batch)

        model.save(model_path, save_format="h5")

    return model


def predict_and_check(previous_prices_list, actual_price_float, model, digits_to_round_to):
    previous_prices_list = np.array(previous_prices_list)
    previous_prices_list = np.reshape(previous_prices_list,
                                      (previous_prices_list.shape[0], previous_prices_list.shape[1], 1))

    prediction = round(model.predict(previous_prices_list)[-1:][0][0], digits_to_round_to)
    difference = round(prediction - actual_price_float, digits_to_round_to)
    percent_error = round((difference / actual_price_float) * 100, digits_to_round_to)
    accuracy = round(100 - abs(percent_error), 2)
    total_accuracies.append(accuracy)
    total_differences.append(difference)

    if accuracy >= 85:
        accuracy_color = Fore.GREEN
    elif accuracy >= 70:
        accuracy_color = Fore.YELLOW
    else:
        accuracy_color = Fore.RED

    average_accuracy = round(np.array(total_accuracies).mean(), digits_to_round_to)
    if average_accuracy >= 85:
        average_accuracy_color = Fore.GREEN
    elif average_accuracy >= 70:
        average_accuracy_color = Fore.YELLOW
    else:
        average_accuracy_color = Fore.RED

    actual_price_str = str(actual_price_float)

    print(f'{Fore.WHITE}+------------------------------------+')
    print(f'{Fore.WHITE}Newest prediction was: {Fore.LIGHTBLUE_EX}{str(prediction)}')
    print(f'{Fore.WHITE}Actual price was: {Fore.GREEN}{actual_price_str}')
    print(f'{Fore.WHITE}Difference was: {Fore.LIGHTRED_EX}{difference}')
    print(f'{Fore.WHITE}Mean Difference is: {Fore.LIGHTRED_EX}{round(np.array(total_differences).mean(), digits_to_round_to)}')
    print(f'{Fore.WHITE}Accuracy was: {accuracy_color}{accuracy}')
    print(f'{Fore.WHITE}Mean Accuracy is: {average_accuracy_color}{average_accuracy}')
    print(f'{Fore.WHITE}Over {len(total_differences)} predictions for the current session')
    print(f'{Fore.WHITE}+------------------------------------+')


def train_on_batched_live_data(live_data, model, model_path, number_to_batch):
    print(f'XXXXXXXXX Re-training on past 300 live data prices XXXXXXXXX')

    live_data = np.array(live_data)

    x_train = []
    y_train = []
    for i in range(5, 300):
        x_train.append(live_data[i - 5:i, 0])
        y_train.append(live_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model.fit(x_train, y_train, batch_size=number_to_batch, epochs=100, shuffle=True, verbose=0)
    model.save(model_path, save_format="h5")
    print('XXXXXXXXX Finished training! XXXXXXXXX')
