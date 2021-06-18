import time

import numpy as np
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

        if len(previous_prices) % number_to_batch == 0:
            train_on_batched_live_data(previous_prices, open_price, number_to_batch, model)
            print(f'XXXXXXXXX Re-training on past {number_to_batch} live data prices XXXXXXXXX')
            time.sleep(3)
            previous_prices = []

        time.sleep(1)
        loop_iterator += 1
