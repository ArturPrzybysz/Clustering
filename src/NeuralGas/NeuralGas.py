from NeuralGas.Neuron import Neuron
import numpy as np


class NeuralGas:
    def __init__(self, neuron_count, input_size, initial_learning_rate, final_learning_rate, initial_neighborhood_coef,
                 final_neighborhood_coef):
        self.neurons = self._init_neurons(neuron_count, input_size)
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.learning_rate = initial_learning_rate

    @staticmethod
    def _init_neurons(neuron_count, input_size):
        return [Neuron(input_size) for i in range(0, neuron_count)]

    def learn(self, data, epochs):
        for e in range(0, epochs):
            self._update_learning_rate(e, epochs)
            np.random.shuffle(data)
            for d in data:
                self.learn_from_vector(d)

    def _update_learning_rate(self, current_epoch, max_epoch):
        self.learning_rate = self.initial_learning_rate * pow(self.final_learning_rate / self.initial_learning_rate,
                                                              current_epoch / max_epoch)

    def learn_from_vector(self, vector):
        self._update_neurons_distance(vector)
        self._sort_neurons_by_distance()
        self._apply_change_to_neurons(vector)

    def _apply_change_to_neurons(self, vector):
        pass

    def _update_neurons_distance(self, vector):
        for n in self.neurons:
            n.update_distance(vector)

    def _sort_neurons_by_distance(self):
        self.neurons.sort(key=lambda x: x.distance, reverse=True)
