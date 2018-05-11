import numpy as np


def scale_data_set_to_range(data_set, range_begin: float, range_end: float):
    dimensions = len(data_set[0])
    for dim in range(0, dimensions):
        min_value = data_set[0][dim]
        max_value = data_set[0][dim]

        for i in range(0, len(data_set) - 1):
            if data_set[i][dim] < min_value:
                min_value = data_set[i][dim]
            if data_set[i][dim] > max_value:
                max_value = data_set[i][dim]

        for i in range(0, len(data_set) - 1):
            data_set[i][dim] = range_begin + (range_begin - range_end) * (data_set[i][dim] - min_value) / (
                    min_value - max_value)

    return data_set


# TODO change
def generate_triangle():
    sample = np.random.rand(3000, 2)
    sample_filtered = []
    for s in sample:
        if s[0] < 2 * s[1] - 1:
            sample_filtered.append(s)
    return sample_filtered


def generate_heart():
    sample = np.random.rand(3000, 2)
    sample_filtered = []

    for s in sample:
        x = s[0] * 2.5 - 1.2
        y = s[1] * 2.5 - 1.5
        if (x ** 2 + y ** 2 - 1) ** 3 + (x ** 2) * (y ** 3) <= 0:
            sample_filtered.append(s)
    return sample_filtered


# TODO implement
def read_file(filename):
    pass