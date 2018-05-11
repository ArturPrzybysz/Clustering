import printer
import data_import

from SelfOrganizingMap.NeighborhoodFunction.GaussianFunction import GaussianFunction
from SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap

neighborhoodFunction = GaussianFunction(radius=4)
som = SelfOrganizingMap(matrix_height=12,
                        matrix_width=12,
                        input_length=2,
                        neighborhood_function=neighborhoodFunction,
                        learning_rate=0.65,
                        minimum_tiredness_potential=0.75)

data = data_import.generate_heart()

data = data_import.scale_data_set_to_range(data, 0, 1)

printer.print_neurons_connections_over_data_points(som, data, width=900, height=900)

som.learn(data, 30)

# activation = som.activation_count_map(sample)
# print(activation)

printer.print_neurons_connections_over_data_points(som, data, width=900, height=900)
