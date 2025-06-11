import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # přidání cesty k rodičovskému adresáři
from preprocessing import prepare_data

# This function prepares the data for testing with a limited number of rows.


def prepare_data_limited(n_rows=5000):
    df, X_train, X_test, y_train, y_test = prepare_data()

    X_train = X_train.head(n_rows)
    y_train = y_train.head(n_rows)
    X_test = X_test.head(n_rows)
    y_test = y_test.head(n_rows)
    df = df.head(n_rows * 2)  # aby df nemělo původní velikost

    return df, X_train, X_test, y_train, y_test
