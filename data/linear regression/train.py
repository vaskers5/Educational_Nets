import pandas as pd
import argparse
from utils.model import LinearRegression
import numpy as np
import yaml


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
        Python script to start train loop for linear regression
     Examples:
        1)python3 train.py --config data/config_name
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--config", "config", type=str, dest="d", required=True, help="Path to config")
    return parser.parse_args()


def mse(predict: np.array, labels: np.array) -> float:
    dif = np.square(predict - labels)
    return dif.sum()/(2 * len(predict))


class InvalidConfigException(Exception):
    pass


if __name__ == "__main__":
    args = vars(parse_args())
    df = pd.read_csv(args['train_data']).select_dtypes(include=[int, float]).dropna()
    with open(args['config'], 'r') as f:
        params = yaml.download(f)

    model = LinearRegression(mse)
    print(model.train(df.drop(params['target'], 1),
          df[params['target']], **params['train'])[-1])
    if params['mode'] == "train":
        model.save_weights(args['model_path'])
    else:
        raise InvalidConfigException("Use test config to train Net")
