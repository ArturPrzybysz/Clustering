from src.LearningData import LearningData
from src.SelfOrganizingMap.NeighborhoodFunction.NeighborhoodFunction import NeighborhoodFunction
import random

from src.SelfOrganizingMap.Neuron import Neuron
from src.util.MathUtil import euclidean_distance


class SelfOrganizingMap:

    def __init__(self, matrix_height: int, matrix_width: int, dimension: int,
                 neighborhood_function: NeighborhoodFunction):
        self.neurons = self.init_neurons(matrix_height, matrix_width, dimension)
        self.matrix_height = matrix_height
        self.matrix_width = matrix_width
        self.neighborhood_function = neighborhood_function

    @staticmethod
    def init_neurons(matrix_height: int, matrix_width: int, dimension: int):
        return [[Neuron(dimension, x, y) for x in range(0, matrix_width)] for y in range(0, matrix_height)]

    def learn(self, data: LearningData, epochs):
        for e in epochs:
            random.shuffle(data)
            for d in data.data:
                closestNeuron: Neuron = self.find_closest_neuron(self, d)
                self.neighborhood_function.apply(self, closestNeuron, 1)

    @staticmethod
    def find_closest_neuron(self, data):
        closest_neuron: Neuron = self.neurons[0][0]
        smallest_distance = euclidean_distance(closest_neuron, data)

        for neuronsRow in self.neurons:
            for neuron in neuronsRow:
                distance = euclidean_distance(neuron, data)
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_neuron = neuron

        return closest_neuron
