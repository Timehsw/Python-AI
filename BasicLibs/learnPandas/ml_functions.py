# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/23
    Desc : 
    Note : 
'''

import numpy as np
import pandas as pd
import scipy.spatial
import math

"""Singular Value Decomposition Functions"""


# Reduce to k principal components via singular value decomposition
# data is a (m x n) dataframe of data, k is the final dimensions
def svd_reduce(df_data, k, df_countries):
    # Normalize data
    df_data_normalized = (df_data - df_data.mean()) / df_data.std()

    # Convert to numpy array
    x = df_data_normalized.as_matrix()

    # Performs singular value decomposition
    u, s, v_transpose = np.linalg.svd(x)

    # Reduce to principal components
    x_transpose = np.transpose(x)
    us_transpose = np.dot(v_transpose, x_transpose)
    us = np.transpose(us_transpose)

    df_principal_components = pd.DataFrame(data=us[:, range(0, k)],
                                           index=df_countries,
                                           columns=list(map(str, range(0, k))))

    return df_principal_components


"""K-Means Functions"""


def k_means(df_data, k):

    # Choose k random indices
    indices = df_data.index.values.copy()
    np.random.shuffle(indices)  # shuffle all indices randomly
    random_indices = indices[0:k]

    # Use chosen random points to initialize means
    means = df_data.loc[random_indices].as_matrix()

    np_data = df_data.as_matrix()

    iterations = 0
    converged = False
    while not converged:
        iterations += 1

        # Initialize new means
        new_means = np.zeros(means.shape)
        bs = [0] * k

        # For each x
        for x in np_data:
            # check the distance to each mean
            distances = []
            for mean in means:
                distances.append(scipy.spatial.distance.minkowski(x, mean, 2))

            # add x to the appropriate new mean (closest)
            new_means[distances.index(min(distances))] += x
            bs[distances.index(min(distances))] += 1

        # Computer means
        for i, (b, new_mean) in enumerate(zip(bs, new_means)):
            new_means[i] = new_mean / (b * 1.0)

        # Check if converged
        if np.array_equal(new_means, means):
            converged = True

        # Set means
        means = new_means

    # initialize list of empty list of indices for each cluster
    clusters = [[] for i in range(k)]
    clusters_indices = [[] for i in range(k)]

    # For each x
    for i, x in enumerate(np_data):
        # check the distance to each mean
        distances = []
        for mean in means:
            distances.append(scipy.spatial.distance.minkowski(x, mean, 2))

        # add x to the appropriate new mean (closest)
        closest = distances.index(min(distances))
        clusters[closest].append(x)
        clusters_indices[closest].append(i)

    return means, clusters, clusters_indices, iterations


"""Dunn Index Functions"""


def intercluster_distances(means):
    intercluster_distance = []
    for i, mean in enumerate(means):
        for mean2 in means[i+1:]:
            intercluster_distance.append(scipy.spatial.distance.minkowski(mean, mean2, 2))

    return intercluster_distance


def intracluster_distances(means, clusters):
    intracluster_distance = []
    for i, (cluster, mean) in enumerate(zip(clusters, means)):
        intracluster_distance.append(0)
        for x in cluster:
            intracluster_distance[i] += scipy.spatial.distance.minkowski(x, mean, 2)

        intracluster_distance[i] /= (1.0 * len(cluster))

    return intracluster_distance


def dunn_index(means, clusters):
    minimum = min(intercluster_distances(means))
    maximum = max(intracluster_distances(means, clusters))

    return (1.0 * minimum) / maximum


"""EM Gaussian Functions"""


def em_mean(np_data, cluster_h):
    sum_hx = np.zeros((1, len(np_data[0])))

    for x, h in zip(np_data, cluster_h):
        sum_hx += (h * x)

    return (1.0 * sum_hx) / sum(cluster_h)


def em_covariance_matrix(np_data, cluster_h, mean):
    sum_hc = np.zeros((len(np_data[0]), len(np_data[0])))

    for x, h in zip(np_data, cluster_h):
        diff = np.matrix(x - mean)
        plus_mat = h * np.matmul(np.transpose(diff), diff)
        sum_hc += plus_mat

    return (1.0 * sum_hc) / sum(cluster_h)


def pi_calc(cluster_h):
    return (1.0 * sum(cluster_h)) / len(cluster_h)


def h_calc(x, mean, covariance_matrix, pi):
    diff = np.matrix(x - mean)
    cm_inv = np.linalg.inv(covariance_matrix)
    first = np.matmul(diff, cm_inv)
    second = np.matmul(first, np.transpose(diff))
    exponent = -0.5 * second
    exp = np.exp(exponent)

    cm_det = np.linalg.det(covariance_matrix)
    h = (pi * exp) / math.sqrt(cm_det)
    return h


def em_gaussian(df_data, k):
    df_data = (df_data - df_data.mean()) / df_data.std()
    # Initialization
    means, clusters, clusters_indices, iterations = k_means(df_data, k)

    np_data = df_data.as_matrix()

    # Calculate initial hs
    cluster_hs = np.zeros((len(clusters), len(df_data)))
    for i, cluster_indices in enumerate(clusters_indices):
        for index in cluster_indices:
            cluster_hs[i][index] = 1.0

    # Calculate initial means and covariance matrices
    means = np.zeros((k, len(means[0])))
    covariance_matrices = np.zeros((k, len(means[0]), len(means[0])))
    pis = np.zeros((k, 1))
    for i, cluster_h in enumerate(cluster_hs):
        means[i] = em_mean(np_data, cluster_h)
        covariance_matrices[i] = em_covariance_matrix(np_data, cluster_h, means[i])
        pis[i] = pi_calc(cluster_h)

    # Run until no labels change
    old_labels = np.zeros((len(df_data), 1))
    converged = False
    iterations = 0
    while not converged:
        iterations += 1

        # Expectation
        cluster_hs = np.zeros((len(clusters), len(df_data)))
        for i, x in enumerate(np_data):
            for j, (mean, covariance_matrix, pi) in enumerate(zip(means, covariance_matrices, pis)):
                h_num = h_calc(x, mean, covariance_matrix, pi)
                h_denom = 0
                for kp, (mean2, covariance_matrix2, pi2) in enumerate(zip(means, covariance_matrices, pis)):
                    h_denom += h_calc(x, mean2, covariance_matrix2, pi2)

                cluster_hs[j, i] = (1.0 * h_num) / h_denom

        # Maximization
        for i, cluster_h in enumerate(cluster_hs):
            means[i] = em_mean(np_data, cluster_h)  # Write em_mean
            covariance_matrices[i] = em_covariance_matrix(np_data, cluster_h, means[i])  # Write em_covariance

        # Label and check for convergence
        new_labels = np.zeros((len(df_data), 1))
        for i, column in enumerate(np.transpose(cluster_hs)):
            new_labels[i] = np.argmax(column)

        if np.array_equal(old_labels, new_labels):
            converged = True

        # Reset for next time
        old_labels = new_labels

    return means, old_labels