import math
import numpy as np


def euclidean_distance(a: np.array, b: np.array):
    return np.linalg.norm(a - b)


def gaussian_function(x: float, delta: float):
    return math.exp(-x * x / delta)


def euclidean_normalization(matrix):
    for vector in matrix:
        length = euclidean_distance(vector, np.zeros_like(vector))
        vector /= length
    return matrix
