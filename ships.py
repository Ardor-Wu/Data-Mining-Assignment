import datetime
import sys
import os
import random
from rdp import rdp
import numpy as np
import pandas as pd
import traj_dist.distance as tdist
from sklearn.neighbors import KNeighborsClassifier

train_size, val_size, test_size = 600, 200, 200

lon_adjust_factor_ship = 1 / np.cos(30.526296012793487)


def load_data(directory, size, sample_rate=1000, lon_adjust_factor=1.0):
    indices = random.sample(range(1, 18329 + 1), size)
    Xs, ys = [], []
    for i in indices:
        data = pd.read_csv(os.path.join(directory, str(i) + '.csv').replace('\\', '/'))
        lat = np.array(data.iloc[:, 1])[::sample_rate]
        lon = np.array(data.iloc[:, 2])[::sample_rate]
        lon /= lon_adjust_factor
        X = np.vstack([lon, lat]).T
        y = data.iloc[:, -1][0]
        Xs.append(X)
        ys.append(y)
    return Xs, ys


data_dir = '../data'
train_dir = os.path.join(data_dir, 'train_dataset/train').replace('\\', '/')
test_dir = os.path.join(data_dir, 'test_dataset/test_dataset').replace('\\', '/')

metrics = ["frechet", "sspd", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]


# for metric in metrics:
def test_metric(metric, n=5, eps=(10 ** i for i in range(-4, 2)), g=(float(10 ** i) for i in range(-4, 2)),
                type_d='spherical', lon_adjust_factor=1.0):
    accs = []
    eps = list(eps)
    g = list(g)
    k = int(np.floor(np.sqrt(train_size + val_size)))
    for i in range(n):
        X, y = load_data(train_dir, train_size + val_size + test_size, sample_rate=20)
        assert metric in metrics
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        if metric in metrics[:5]:
            X_train = X[:train_size + val_size]
            y_train = y[:train_size + val_size]
        else:
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
        if metric in metrics[:5]:
            knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
            train_dist = tdist.cdist(X_train, X_train, metric=metric, type_d=type_d)
            knn.fit(train_dist, y_train)
            test_dist = tdist.cdist(X_test, X_train, metric=metric, type_d=type_d)
            y_pred = knn.predict(test_dist)
            accs.append(np.mean(y_pred == y_test))
        else:
            val_accs = []
            if metric == "edr" or metric == "lcss":
                for epsilon in eps:
                    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
                    train_dist = tdist.cdist(X_train, X_train, metric=metric, eps=epsilon, type_d=type_d)
                    knn.fit(train_dist, y_train)
                    val_dist = tdist.cdist(X_val, X_train, metric=metric, eps=epsilon, type_d=type_d)
                    y_pred = knn.predict(val_dist)
                    val_accs.append(np.mean(y_pred == y_val))
                best_epsilon = list(eps)[np.argmax(val_accs)]
                print(val_accs)
                test_dist = tdist.cdist(X_test, X_train + X_val, metric=metric, eps=best_epsilon, type_d=type_d)
                train_dist = tdist.cdist(X_train + X_val, X_train + X_val, metric=metric, eps=best_epsilon,
                                         type_d=type_d)
                knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
                knn.fit(train_dist, y_train + y_val)
                y_pred = knn.predict(test_dist)
                accs.append(np.mean(y_pred == y_test))
            else:  # metric="erp"
                for g_value in g:
                    g_value = np.array([g_value, g_value])
                    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
                    train_dist = tdist.cdist(X_train, X_train, metric=metric, g=g_value, type_d=type_d)
                    knn.fit(train_dist, y_train)
                    val_dist = tdist.cdist(X_val, X_train, metric=metric, g=g_value, type_d=type_d)
                    y_pred = knn.predict(val_dist)
                    val_accs.append(np.mean(y_pred == y_val))
                best_g = list(g)[np.argmax(val_accs)]
                best_g = np.array([best_g, best_g])
                print(val_accs)
                test_dist = tdist.cdist(X_test, X_train + X_val, metric=metric, g=best_g, type_d=type_d)
                train_dist = tdist.cdist(X_train + X_val, X_train + X_val, metric=metric, g=best_g, type_d=type_d)
                knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
                knn.fit(train_dist, y_train + y_val)
                y_pred = knn.predict(test_dist)
                accs.append(np.mean(y_pred == y_test))

    return accs


# print(metrics[2], test_metric(metrics[2], type_d='euclidean'))

# for i in range(1, 7):
#    print(metrics[i], test_metric(metrics[i]))
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('distance_index', metavar='N', type=int, nargs='+',
                    help='an integer distance index')

args = parser.parse_args()
for index in args.distance_index:
    starttime = datetime.datetime.now()
    sys.stdout = open(str(index) + '_ori.txt', "wt")
    print(metrics[index])
    print(test_metric(metrics[index], type_d='euclidean'))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    sys.stdout = open(str(index) + '_lon_adjusted.txt', "wt")
    print(metrics[index])
    print(test_metric(metrics[index], type_d='euclidean', lon_adjust_factor=lon_adjust_factor_ship))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    if metrics[index] != 'discret_frechet':
        sys.stdout = open(str(index) + '_spherical.txt', "wt")
        print(metrics[index])
        print(test_metric(metrics[index]))
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
