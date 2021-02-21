import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that runs before each test

    yield
    # Code that runs after each test

    plt.close()


xs = np.random.rand(100)
y_pred = xs + 0.1 * np.random.normal(size=100)
y_true = xs + 0.1 * np.random.normal(size=100)
