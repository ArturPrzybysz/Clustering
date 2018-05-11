from src.SelfOrganizingMap.NeighborhoodFunction.NeighborhoodFunction import NeighborhoodFunction
from SelfOrganizingMap.Neuron import Neuron
from src.SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap
from src.util.MathUtil import euclidean_distance
from src.util.MathUtil import gaussian_function

import numpy as np


class GaussianFunction(NeighborhoodFunction):

    def __init__(self, radius):
        self.radius = radius

    def apply(self, som: SelfOrganizingMap, winner_neuron: Neuron, learning_rate: float, learning_point: np.array):
        radius = int(self.radius)
        for x in range(0, som.matrix_height):
            for y in range(0, som.matrix_width):
                if (x - winner_neuron.x) ** 2 + (y - winner_neuron.y) ** 2 < radius ** 2:
                    distance = euclidean_distance(som.neurons[x][y].weights, winner_neuron.weights)
                    vector_of_change = learning_point - som.neurons[x][y].weights
                    update_vector = vector_of_change * gaussian_function(distance, 1 / radius) * learning_rate
                    som.neurons[x][y].update(update_vector)

        self.radius -= 0.0001
        if self.radius < 1:
            self.radius = 1
