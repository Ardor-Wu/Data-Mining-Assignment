import datetime
import sys
import os
import random
from rdp import rdp
import numpy as np
import pandas as pd
import traj_dist.distance as tdist
from collective_classification import distance_to_similarity, node_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import scipy

train_size, val_size, test_size = 600, 200, 200

lon_adjust_factor_ship = 1 / np.cos(30.526296012793487)


def load_data(directory, size, sample_rate=1000, lon_adjust_factor=1.0):
    indices = random.sample(range(1, 18329 + 1), size)
    Xs, ys = [], []
    for i in indices:
        data = pd.read_csv(os.path.join(directory, str(i) + '.csv').replace('\\', '/'))
        lat = np.array(data.iloc[:, 1])[::sample_rate]
        lon = np.array(data.iloc[:, 2])[::sample_rate]
        lon *= lon_adjust_factor
        X = np.vstack([lon, lat]).T
        y = data.iloc[:, -1][0]
        Xs.append(X)
        ys.append(y)
    return Xs, ys


data_dir = '../data'
train_dir = os.path.join(data_dir, 'train_dataset/train').replace('\\', '/')
test_dir = os.path.join(data_dir, 'test_dataset/test_dataset').replace('\\', '/')

metrics = ["frechet", "sspd", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]


def KNN_classification(train_dist, test_dist, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', n_jobs=-1)
    knn.fit(train_dist, y_train)
    y_pred = knn.predict(test_dist)
    return np.mean(y_pred == y_test)


def acc(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    correct = 0
    for i in range(len(y_test)):
        if y_test[i][y_pred[i]] == 1:
            correct += 1
    return correct / len(y_test)


def graph_classification(dist, k, y_train, y_test, val=False, mu=None, temp=None):
    label_encoder = OneHotEncoder(sparse=False)
    label_encoder.fit(np.array(y_train).reshape(-1, 1))
    y_train_encoded = label_encoder.transform(np.array(y_train).reshape(-1, 1))
    y_test_encoded = label_encoder.transform(np.array(y_test).reshape(-1, 1))
    y_input = np.vstack([y_train_encoded, np.zeros_like(y_test_encoded)])
    if mu:
        assert temp
        similarity = distance_to_similarity(dist, k, temp)
        y_pred = node_classification(similarity, y_input)[-test_size:]
        return acc(y_pred, y_test_encoded), mu, temp
    else:
        best_mu, best_temp = None, None
        best_accuracy = -np.inf
        for temp in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
            similarity = distance_to_similarity(dist, k, temp)
            if val:
                for mu in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
                    y_pred = node_classification(similarity, y_input)[-val_size:]
                    accuracy = acc(y_pred, y_test_encoded)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_mu = mu
                        best_temp = temp
            else:
                similarity = similarity[:-test_size, :-test_size]  # train and val
                y_input_val = np.vstack([y_input[:train_size], np.zeros((val_size, y_input.shape[1]))])
                for mu in (0.01, 0.1, 1, 10, 100):
                    y_pred = node_classification(similarity, y_input_val, mu)[-val_size:]
                    accuracy = acc(y_pred, y_train_encoded[train_size:])
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_mu = mu
                        best_temp = temp
    if val:
        return best_accuracy, best_mu, best_temp
    else:
        similarity = distance_to_similarity(dist, k, best_temp)
        y_pred = node_classification(similarity, y_input, best_mu)[-test_size:]
        return acc(y_pred, y_test_encoded), best_mu, best_temp


# for metric in metrics:
def test_metric(metric, n=5, eps=(10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1),
                # g=(10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1),
                g=0.0, type_d='spherical', lon_adjust_factor=1.0, algorithm='KNN'):
    accs = []
    eps = list(eps)
    k = int(np.floor(np.sqrt(train_size + val_size)))
    for i in range(n):
        X, y = load_data(train_dir, train_size + val_size + test_size, sample_rate=20,
                         lon_adjust_factor=lon_adjust_factor)
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
            if algorithm == 'KNN':
                train_dist = tdist.cdist(X_train, X_train, metric=metric, type_d=type_d)
                test_dist = tdist.cdist(X_test, X_train, metric=metric, type_d=type_d)
                accs.append(KNN_classification(train_dist, test_dist, y_train, y_test, k))
            else:
                dist = tdist.cdist(X_train + X_test, X_train + X_test, metric=metric, type_d=type_d)
                res = graph_classification(dist, k, y_train, y_test)
                print('best_mu:', res[1])
                print('best_temp:', res[2])
                accs.append(res[0])
        else:
            val_accs = []
            best_mu, best_temp = [], []
            if metric == "edr" or metric == "lcss":
                for epsilon in eps:
                    if algorithm == 'KNN':
                        train_dist = tdist.cdist(X_train, X_train, metric=metric, eps=epsilon, type_d=type_d)
                        val_dist = tdist.cdist(X_val, X_train, metric=metric, eps=epsilon, type_d=type_d)
                        val_accs.append(KNN_classification(train_dist, val_dist, y_train, y_val, k))
                    else:
                        dist = tdist.cdist(X_train + X_val, X_train + X_val, metric=metric, type_d=type_d,eps=epsilon)
                        res = graph_classification(dist, k, y_train, y_val, val=True)
                        val_accs.append([res[0]])
                        best_mu.append(res[1])
                        best_temp.append(res[2])
                best = np.argmax(val_accs)
                best_epsilon = list(eps)[best]
                if algorithm != 'KNN':
                    best_mu = best_mu[best]
                    best_temp = best_temp[best]
                print(val_accs)
                if algorithm == 'KNN':
                    test_dist = tdist.cdist(X_test, X_train + X_val, metric=metric, eps=best_epsilon, type_d=type_d)
                    train_dist = tdist.cdist(X_train + X_val, X_train + X_val, metric=metric, eps=best_epsilon,
                                             type_d=type_d)
                    accs.append(KNN_classification(train_dist, test_dist, y_train + y_val, y_test, k))
                else:
                    dist = tdist.cdist(X_train + X_val + X_test, X_train + X_val + X_test, metric=metric, type_d=type_d,
                                       eps=best_epsilon)
                    res = graph_classification(dist, k, y_train + y_val, y_test, mu=best_mu, temp=best_temp)
                    accs.append(res[0])
                    best_mu = res[1]
                    best_temp = res[2]
                    print('best_mu:', best_mu)
                    print('best_temp:', best_temp)
            else:  # metric="erp"
                best_g = np.array([g, g])
                if algorithm == 'KNN':
                    test_dist = tdist.cdist(X_test, X_train + X_val, metric=metric, g=best_g, type_d=type_d)
                    train_dist = tdist.cdist(X_train + X_val, X_train + X_val, metric=metric, g=best_g, type_d=type_d)
                    accs.append(KNN_classification(train_dist, test_dist, y_train + y_val, y_test, k))
                else:
                    dist = tdist.cdist(X_train + X_val + X_test, X_train + X_val + X_test, metric=metric, type_d=type_d,
                                       g=best_g)
                    res = graph_classification(dist, k, y_train + y_val, y_test)
                    print('best_mu:', res[1])
                    print('best_temp:', res[2])
                    accs.append(res[0])

    return accs


# print(metrics[2], test_metric(metrics[2], type_d='euclidean'))

# for i in range(1, 7):
#    print(metrics[i], test_metric(metrics[i]))
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--algorithm', default='KNN', type=str,
                    help='KNN or graph')
parser.add_argument('--d_type', default=0, type=int, metavar='N')
parser.add_argument('distance_index', metavar='N', type=int, nargs='+',
                    help='an integer distance index')

args = parser.parse_args()
for index in args.distance_index:
    starttime = datetime.datetime.now()
    if args.d_type == 0:
        sys.stdout = open(str(index) + '_ori_' + args.algorithm + '.txt', "wt")
    elif args.d_type == 1:
        sys.stdout = open(str(index) + '_lon_adjusted_' + args.algorithm + '.txt', "wt")
    elif args.d_type == 2:
        if metrics[index] != 'discret_frechet':
            sys.stdout = open(str(index) + '_spherical_' + args.algorithm + '.txt', "wt")
    print(metrics[index])
    if args.d_type == 0:
        print(test_metric(metrics[index], type_d='euclidean', algorithm=args.algorithm))
    elif args.d_type == 1:
        print(test_metric(metrics[index], type_d='euclidean', lon_adjust_factor=lon_adjust_factor_ship,
                          algorithm=args.algorithm))
    elif args.d_type == 2:
        if metrics[index] != 'discret_frechet':
            print(test_metric(metrics[index], algorithm=args.algorithm))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
