from src.SelfOrganizingMap.NeighborhoodFunction.NeighborhoodFunction import NeighborhoodFunction
from src.SelfOrganizingMap.Neuron import Neuron
from src.SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap
from src.util.MathUtil import euclidean_distance
from src.util.MathUtil import gaussian_function


class GaussianFunction(NeighborhoodFunction):

    def __init__(self, radius):
        self.radius = radius

    # TODO CHANGE LEARNING RATE OVER TIME
    # TODO ADD NEURONS TIRING
    def apply(self, som: SelfOrganizingMap, winner_neuron: Neuron, learning_rate: float):
        X = int(self.radius)
        for x in range(-X, X + 1):
            Y = int((self.radius * self.radius - x * x) ** 0.5)
            for y in range(-Y, Y + 1):
                if (x >= 0 & x < som.matrix_width) & (
                        y >= 0 & y < som.matrix_height):
                    distance = euclidean_distance(som.neurons[x][y].weights, winner_neuron.weights)
                    update_value = gaussian_function(distance) * learning_rate
                    som.neurons[x][y].update(update_value)
