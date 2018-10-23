# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/23
    Desc : 
    Note : 
'''


import matplotlib.pyplot as plt
from BasicLibs.learnPandas.ml_functions import svd_reduce, k_means, dunn_index


def part2(df_deaths, df_countries):
    # Reduce to k=5 principal components
    df_principle_components = svd_reduce(df_deaths, 5, df_countries)

    # Reduce to k=2 principal components
    df_graphing_components = svd_reduce(df_deaths, 2, df_countries)

    # cluster normal data with different cluster numbers and decide best
    print("k vs dunn index for full data with K-means")
    for k in range(2, 10):
        # Perform clustering
        means, clusters, cluster_indices, iterations = k_means(df_deaths, k)

        # Calculate dunn_index for clustering job, bigger is better
        dunn = dunn_index(means, clusters)
        print(k, dunn)
    print("")

    # label best clusters in the 2-principal-component graph : 2
    k = 2
    means, clusters, cluster_indices, iterations = k_means(df_deaths, k)

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    graph_clusters = [[] for i in range(k)]
    for i in range(k):
        graph_clusters[i] = df_graphing_components.iloc[cluster_indices[i]]
        plt.scatter(graph_clusters[i]['0'], graph_clusters[i]['1'], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/km_normal_data_2_cluster_plot.png", facecolor='white')

    # Cluster but using 5 pc reduced data
    print("k vs dunn index for 5 principle component data with K-means")
    for k in range(2, 10):
        # Perform clustering
        means, clusters, cluster_indices, iterations = k_means(df_principle_components, k)

        # Calculate dunn_index for clustering job, bigger is better
        dunn = dunn_index(means, clusters)
        print(k, dunn)
    print("")

    # label best clusters in the 2-principal-component graph : 2
    k = 2
    means, clusters, cluster_indices, iterations = k_means(df_principle_components, k)

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    graph_clusters = [[] for i in range(k)]
    for i in range(k):
        graph_clusters[i] = df_graphing_components.iloc[cluster_indices[i]]
        plt.scatter(graph_clusters[i]['0'], graph_clusters[i]['1'], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/km_5pc_data_2_cluster_plot.png", facecolor='white')

    # Cluster but using 2 pc reduced data
    print("k vs dunn index for 2 principle component data with K-means")
    for k in range(2, 10):
        # Perform clustering
        means, clusters, cluster_indices, iterations = k_means(df_graphing_components, k)

        # Calculate dunn_index for clustering job, bigger is better
        dunn = dunn_index(means, clusters)
        print(k, dunn)
    print("")

    # label best clusters in the 2-principal-component graph : 2, 4
    k = 2
    means, clusters, cluster_indices, iterations = k_means(df_graphing_components, k)

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    graph_clusters = [[] for i in range(k)]
    for i in range(k):
        graph_clusters[i] = df_graphing_components.iloc[cluster_indices[i]]
        plt.scatter(graph_clusters[i]['0'], graph_clusters[i]['1'], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/km_2pc_data_2_cluster_plot.png", facecolor='white')

    k = 4
    means, clusters, cluster_indices, iterations = k_means(df_graphing_components, k)

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    graph_clusters = [[] for i in range(k)]
    for i in range(k):
        graph_clusters[i] = df_graphing_components.iloc[cluster_indices[i]]
        plt.scatter(graph_clusters[i]['0'], graph_clusters[i]['1'], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/km_2pc_data_4_cluster_plot.png", facecolor='white')