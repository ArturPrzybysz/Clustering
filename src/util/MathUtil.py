import math


def euclidean_distance(A, B):
    inner_sum = 0
    for a, b in zip(A, B):
        inner_sum += pow(a - b, 2)
    return pow(inner_sum, 0.5)


def gaussian_function(x: float):
    return math.exp(-0.5 * x * x)
