import printer
import data_import
import numpy as np
from NeuralGas.NeuralGas import NeuralGas
import matplotlib.pyplot as plt

neural_gas = NeuralGas(neuron_count=40,
                       input_size=2,
                       initial_learning_rate=3,
                       final_learning_rate=0.3,
                       initial_neighborhood_coef=5,
                       final_neighborhood_coef=0.1)

dataA = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.2, 0.4)
dataB = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.4, 0.5)
dataC = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.6, 0.75)
dataD = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.8, 1)

data = np.concatenate((dataA, dataB, dataC, dataD))

data = data[:1000]

neural_gas.learn(data, epochs=10)

printer.print_neurons_over_data_points(neurons=neural_gas.neurons, data=data, height=900, width=900)

plt.hist(neural_gas.quantization_errors, bins=len(neural_gas.quantization_errors))
plt.show()
