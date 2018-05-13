import printer
import data_import

from NeuralGas.NeuralGas import NeuralGas

neural_gas = NeuralGas(neuron_count=40,
                       input_size=2,
                       initial_learning_rate=3,
                       final_learning_rate=0.3,
                       initial_neighborhood_coef=5,
                       final_neighborhood_coef=0.1)

data = data_import.generate_heart()
data = data_import.scale_data_set_to_range(data, 0, 0.1)

neural_gas.learn(data, epochs=10)

printer.print_neurons_over_data_points(neurons=neural_gas.neurons, data=data, height=900, width=900)
