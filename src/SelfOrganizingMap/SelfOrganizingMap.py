import numpy as np

from SelfOrganizingMap.NeighborhoodFunction.NeighborhoodFunction import NeighborhoodFunction
from SelfOrganizingMap.Neuron import Neuron
from util.MathUtil import euclidean_distance
from printer import save_neurons_connections_over_data_points


class SelfOrganizingMap:

    def __init__(self, matrix_height: int, matrix_width: int, input_length: int,
                 neighborhood_function: NeighborhoodFunction, learning_rate: float,
                 minimum_tiredness_potential: float):
        self.neurons = self._init_neurons(matrix_height, matrix_width, input_length)
        self.matrix_height = matrix_height
        self.matrix_width = matrix_width
        self.neighborhood_function = neighborhood_function
        self.learning_rate = learning_rate
        self.minimum_tiredness_potential = minimum_tiredness_potential

        self._update_active_neurons()

    @staticmethod
    def _init_neurons(matrix_height: int, matrix_width: int, dimension: int):
        return [[Neuron(dimension, x, y) for x in range(0, matrix_width)] for y in range(0, matrix_height)]

    def _update_neurons_tiredness(self, winner: Neuron):
        for x in range(0, self.matrix_height):
            for y in range(0, self.matrix_width):
                if self.neurons[x][y] == winner:
                    winner.tiredness_potential -= self.minimum_tiredness_potential
                else:
                    self.neurons[x][y].tiredness_potential += 0.1
                    if self.neurons[x][y].tiredness_potential > 1:
                        self.neurons[x][y].tiredness_potential = 1

    def _update_active_neurons(self):
        active_neurons = []
        for x in range(0, self.matrix_height):
            for y in range(0, self.matrix_width):
                if self.neurons[x][y].tiredness_potential > self.minimum_tiredness_potential:
                    active_neurons.append(self.neurons[x][y])

        if len(active_neurons) == 0:
            for x in range(0, self.matrix_height):
                for y in range(0, self.matrix_width):
                    self.neurons[x][y].tiredness_potential = 1
                    active_neurons.append(self.neurons[x][y])

        self.active_neurons = active_neurons

    def learn(self, data, epochs: int):
        counter = 0
        for e in range(0, epochs):
            print('Epoch = ', e)
            np.random.shuffle(data)
            learning_speed = self.learning_rate / (1 + e / epochs)
            for d in data:
                if counter % 100 == 0:
                    save_neurons_connections_over_data_points(self, data, str(counter))
                counter += 1

                closest_neuron: Neuron = self.find_closest_active_neuron(self, d)
                self.neighborhood_function.apply(self, closest_neuron, learning_speed, d)
                self._update_neurons_tiredness(closest_neuron)
                self._update_active_neurons()

    @staticmethod
    def find_closest_active_neuron(self, data):
        closest_neuron = self.active_neurons[0]
        smallest_distance = euclidean_distance(closest_neuron.weights, data)

        for n in self.active_neurons:
            distance = euclidean_distance(n.weights, data)
            if distance < smallest_distance:
                closest_neuron = n
                smallest_distance = distance
        return closest_neuron

    def activation_count_map(self, data):
        activation = np.zeros((self.matrix_width, self.matrix_height))
        for d in data:
            winner = self.find_closest_active_neuron(self, d)
            activation[winner.x][winner.y] += 1
        return activation

    def distance_map(self):
        pass
