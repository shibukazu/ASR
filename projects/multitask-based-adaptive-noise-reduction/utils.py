import numpy as np


def calc_slope(x, y):
    assert len(x) == len(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    xy_cov = np.sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
    x_var = np.sum([(x[i] - x_mean) ** 2 for i in range(len(x))])

    return xy_cov / x_var
