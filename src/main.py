import numpy as np
import printer

from util.MathUtil import euclidean_normalization
from SelfOrganizingMap.NeighborhoodFunction.GaussianFunction import GaussianFunction
from SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap

# TODO split drawing points and connections
# TODO save pictures -> make gif of all changes in neurons

neighborhoodFunction = GaussianFunction(6)
som = SelfOrganizingMap(matrix_height=40,
                        matrix_width=40,
                        input_length=2,
                        neighborhood_function=neighborhoodFunction,
                        learning_rate=0.55,
                        minimum_tiredness_potential=0.75)

sample = np.random.rand(5000, 50)

sample_filtered = []
# for s in sample:
#     if s[0] < 2 * s[1] - 1:
#         sample_filtered.append(s)
for s in sample:
    x = s[0] * 2.5 - 1.2
    y = s[1] * 2.5 - 1.5
    if (x ** 2 + y ** 2 - 1) ** 3 + (x ** 2) * (y ** 3) <= 0:
        sample_filtered.append(s)

# sample = euclidean_normalization(sample_filtered)
sample = sample_filtered

height = 900
width = 900
printer.print_neurons_connections_over_data_points(som, sample, width, height)

som.learn(sample, 2)

# activation = som.activation_count_map(sample)
# print(activation)

printer.print_neurons_connections_over_data_points(som, sample, width, height)
