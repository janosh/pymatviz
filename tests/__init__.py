import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ml_matrics import ROOT


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that runs before each test

    yield
    # Code that runs after each test

    plt.close()


y_binary, y_proba, y_clf = pd.read_csv(f"{ROOT}/data/rand_clf.csv").to_numpy().T
xs, y_pred, y_true = pd.read_csv(f"{ROOT}/data/rand_regr.csv").to_numpy().T
