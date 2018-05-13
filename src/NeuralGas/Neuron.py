import numpy as np
import util.MathUtil as util


class Neuron:
    def __init__(self, dimension):
        self.weights = np.random.rand(dimension)
        self.distance: float = 0

    def update_distance(self, vector):
        self.distance = util.euclidean_distance(self.weights, vector)

    def update_weights(self, update_vector):
        self.weights += update_vector
