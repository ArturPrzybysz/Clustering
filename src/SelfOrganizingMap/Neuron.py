import random


class Neuron:
    weights = []
    x = 0
    y = 0

    def __init__(self, dimension: int, x: int, y: int):
        self.weights: [float] = []
        for i in range(0, dimension):
            self.weights[i] = random.uniform(-1, 1)

        self.x: int = x
        self.y: int = y

    def update(self, value: float):
        for neuron in self.weights:
            neuron += value
