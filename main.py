from BaseModel import start_model
from LiveData import collect_live_data


def main():
    print("Predict for DOGE or LITE?")
    # answer = str(input())
    has_correct_answer = False

    # while not has_correct_answer:
    #     if answer.lower() == "doge":
    #         has_correct_answer = True
    model, eval_features, eval_labels = start_model('Resources/DOGE-USD.csv')

    collect_live_data(model, 'https://api.pro.coinbase.com/products/DOGE-USD/candles?granularity=60')
        # elif answer.lower() == "lite":
        #     has_correct_answer = True
        #     model, eval_features, eval_labels = start_model('Resources/LTC-USD.csv')
        #
        #     collect_live_data(model, 'https://chain.so/api/v2/get_price/LTC/USD')
        # else:
        #     print("Incorrect. Please type DOGE or LITE.")
        #     answer = str(input())


if __name__ == "__main__":
    main()
