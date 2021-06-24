import time

import numpy as np
import pandas as pd
import requests

from BaseModel import predict_and_check, train_on_batched_live_data


def get_new_price(url):
    response = requests.get(url).json()
    price_history = pd.DataFrame(response, columns=['Timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
    price_history.drop('Timestamp', 1, inplace=True)

    return price_history


def collect_live_data(model, api_url, model_path, digits_to_round_to, number_to_batch):
    while True:
        price_history = get_new_price(api_url)
        last_price_history = np.array(price_history)[-1:][0]
        new_closing_price = last_price_history[3]

        predict_and_check(price_history, new_closing_price, model, digits_to_round_to)

        train_on_batched_live_data(price_history, model, model_path, number_to_batch)

        time.sleep(60)
