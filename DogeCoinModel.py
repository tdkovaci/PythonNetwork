from BaseModel import start_model, plot_predictions, collect_live_data


def main():
    model, eval_features, eval_labels = start_model('Resources/DOGE-USD.csv')

    plot_predictions(model, eval_features, eval_labels)
    collect_live_data(model, 'https://chain.so/api/v2/get_price/DOGE/USD')


if __name__ == "__main__":
    main()
