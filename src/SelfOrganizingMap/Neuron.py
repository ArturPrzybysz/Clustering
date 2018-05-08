import numpy as np


class Neuron:

    def __init__(self, dimension: int, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.tiredness_potential: float = 1

        self.weights = np.random.rand(dimension)

    def update(self, vector_of_change: np.array):
        for i in range(0, len(self.weights)):
            self.weights[i] += vector_of_change[i]
