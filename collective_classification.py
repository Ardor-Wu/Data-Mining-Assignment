import numpy as np


def distance_to_similarity(distance_mat, k, temp=10):
    """
        distance_mat: N x N
        k: build with kNN neighbors 
        temp: similarity = exp(-d^2/t^2)
    """
    k_min = np.argpartition(distance_mat, k)[:, :k]

    graph = np.zeros_like(distance_mat)
    i_indices = np.array([i for i in range(distance_mat.shape[0]) for _ in range(k)])
    j_indices = k_min.flatten()
    graph[i_indices, j_indices] = 1

    similarity = np.exp(-(distance_mat * distance_mat) / (temp * temp))
    similarity[(graph == 0) | (graph.T == 0)] = 0
    return similarity


def node_classification(weights, labels, mu=1):
    """
        weights: N x N
        labels: N x C
        mu: O_s + mu * O_f
    """
    n = weights.shape[0]
    if 0 in weights.sum(axis=1):
        assert 0
    lambda_mat = np.diag(1 / np.sqrt(weights.sum(axis=1)))
    S = np.matmul(np.matmul(lambda_mat, weights), lambda_mat)
    Z = (mu / (1 + mu)) * np.matmul(np.linalg.inv(np.eye(n) - S / (1 + mu)), labels)
    return np.argmax(Z, axis=1)
