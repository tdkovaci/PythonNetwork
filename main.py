import logging

from BaseModel import start_model
from LiveData import collect_live_data
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')


def main():
    print("Predict for DOGE or LITE?")
    answer = str(input())
    has_correct_answer = False

    model_path = ""
    api_url = ""
    digits_to_round_to = 2

    while not has_correct_answer:
        if answer.lower() == "doge":
            model_path = 'Resources/doge_model'
            api_url = 'https://api.pro.coinbase.com/products/DOGE-USD/candles?granularity=60'
            digits_to_round_to = 4
            break
        elif answer.lower() == "lite":
            model_path = 'Resources/lite_model'
            api_url = 'https://api.pro.coinbase.com/products/LTC-USD/candles?granularity=60'
            digits_to_round_to = 2
            break
        else:
            print("Incorrect. Please type DOGE or LITE.")
            answer = str(input())

    model = start_model(model_path)
    collect_live_data(model, api_url, model_path, digits_to_round_to)


if __name__ == "__main__":
    main()
