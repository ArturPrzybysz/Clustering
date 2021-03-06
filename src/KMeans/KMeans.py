import numpy as np
import printer

from util import MathUtil
from util.MathUtil import euclidean_distance


def k_means(data, cluster_count, epochs, dimensions):
    centroids = data[np.random.choice(np.arange(len(data)), cluster_count), :]
    clusters = _assign_data_to_clusters(data, centroids)
    printer.save_clusters_and_centroids(clusters, centroids, width=900, height=900, filename=str(0))
    quantization_errors = []

    for e in range(0, epochs):
        print('Epoch: ', e)
        clusters = _assign_data_to_clusters(data, centroids)
        centroids = _update_centroids(clusters, centroids)
        printer.save_clusters_and_centroids(clusters, centroids, width=900, height=900, filename=str(e + 1))
        quantization_errors.append(_calculate_quantization(clusters, centroids))

    return clusters, quantization_errors


def _calculate_quantization(clusters, centroids):
    summed_errors = 0
    counter = 0
    for i in range(len(centroids)):
        for c in clusters[i]:
            summed_errors += euclidean_distance(centroids[i], c) ** 2
            counter += 1
    return summed_errors / counter


def _update_centroids(clusters: [np.ndarray], previous_centroid: np.ndarray):
    centroids = np.zeros_like(previous_centroid)
    for c in range(len(clusters)):
        if len(clusters[c]) == 0:
            centroids[c] = previous_centroid[c]
        else:
            centroids[c] = np.mean(clusters[c], axis=0)
    return centroids


def _assign_data_to_clusters(data, centroids):
    cluster_count = len(centroids)
    clusters = [[] for i in range(cluster_count)]

    for d in data:
        distances = np.zeros(cluster_count)

        for i in range(0, cluster_count):
            distances[i] = MathUtil.euclidean_distance(d, centroids[i])

        i_min = np.argmin(distances)
        clusters[i_min].append(d)

    return clusters


def _init_clusters(cluster_count):
    return [[] for i in range(0, cluster_count)]
