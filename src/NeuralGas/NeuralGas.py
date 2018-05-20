import printer
from NeuralGas.Neuron import Neuron
import numpy as np
from util.MathUtil import euclidean_distance


class NeuralGas:
    def __init__(self, neuron_count, input_size, initial_learning_rate, final_learning_rate, initial_neighborhood_coef,
                 final_neighborhood_coef):
        self.neurons = self._init_neurons(neuron_count, input_size)

        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.learning_rate = initial_learning_rate

        self.initial_neighborhood_coefficient = initial_neighborhood_coef
        self.final_neighborhood_coefficient = final_neighborhood_coef
        self.neighborhood_coefficient = initial_neighborhood_coef
        self.quantization_errors = []

    def _calculate_quantization_error(self, data):
        summed_errors = 0
        for d in data:
            self._update_neurons_distance(d)
            self._sort_neurons_by_distance()
            summed_errors += euclidean_distance(d, self.neurons[0].weights) ** 2
        return summed_errors / len(data)

    @staticmethod
    def _init_neurons(neuron_count, input_size):
        return [Neuron(input_size) for i in range(0, neuron_count)]

    def learn(self, data, epochs):
        for e in range(0, epochs):
            self.quantization_errors.append(self._calculate_quantization_error(data))
            printer.save_neurons_over_data_points(self.neurons, data, height=900, width=900, filename=str(e))
            print('Epoch: ', e)
            self._update_learning_rate(e, epochs)
            self._update_neighborhood_coefficient(e, epochs)
            np.random.shuffle(data)
            for d in data:
                self.learn_from_vector(d)
        self.quantization_errors.append(self._calculate_quantization_error(data))

        printer.save_neurons_over_data_points(self.neurons, data, height=900, width=900, filename=str(epochs))

    def _get_learning_speed(self, index):
        return np.exp(-index / self.neighborhood_coefficient)

    def _update_neighborhood_coefficient(self, current_epoch, max_epoch):
        self.neighborhood_coefficient = self.initial_neighborhood_coefficient * pow(
            self.final_neighborhood_coefficient / self.initial_neighborhood_coefficient,
            current_epoch / max_epoch)

    def _update_learning_rate(self, current_epoch, max_epoch):
        self.learning_rate = self.initial_learning_rate * pow(self.final_learning_rate / self.initial_learning_rate,
                                                              current_epoch / max_epoch)

    def learn_from_vector(self, vector):
        self._update_neurons_distance(vector)
        self._sort_neurons_by_distance()
        self._apply_change_to_neurons(vector)

    def _apply_change_to_neurons(self, vector):
        for i in range(0, len(self.neurons)):
            self.neurons[i].weights += self._get_learning_speed(i) * self.learning_rate * (
                    vector - self.neurons[i].weights)

    def _update_neurons_distance(self, vector):
        for n in self.neurons:
            n.update_distance(vector)

    def _sort_neurons_by_distance(self):
        self.neurons.sort(key=lambda x: x.distance, reverse=False)
