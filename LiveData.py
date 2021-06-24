import time

import numpy as np
import pandas as pd
import requests

from BaseModel import predict_and_check, train_on_batched_live_data


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


def get_new_price(url):
    response = requests.get(url).json()
    price_history = pd.DataFrame(response, columns=['Timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
    price_history.drop('Timestamp', 1, inplace=True)

    return price_history


def collect_live_data(model, url):
    while True:
        price_history = get_new_price(url)
        last_price_history = np.array(price_history)[-1:][0]
        new_closing_price = last_price_history[3]

        predict_and_check(price_history, new_closing_price, model)

        train_on_batched_live_data(price_history, model)

        time.sleep(60)
