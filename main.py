import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiple_linear_regression import linear_regression


def main():

    data = pd.read_excel('basketball_players.xls')
    data = data.values
    X_train = data[:, [0, 1]]
    y_train = data[:, 2]

    mean = np.ones(X_train.shape[1])
    std = np.ones(X_train.shape[1])
    for i in range(0, X_train.shape[1]):
        mean[i] = np.mean(X_train.transpose()[i])
        std[i] = np.std(X_train.transpose()[i])
        for j in range(0, X_train.shape[0]):
            X_train[j][i] = (X_train[j][i] - mean[i])/std[i]

    theta, cost = linear_regression(X_train, y_train, 0.0001, 300000)

    cost = list(cost)
    n_iterations = [x for x in range(1, 300001)]
    plt.plot(n_iterations, cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')


if __name__ == '__main__':
    main()