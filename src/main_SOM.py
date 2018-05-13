import printer
import data_import

from SelfOrganizingMap.NeighborhoodFunction.GaussianFunction import GaussianFunction
from SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap

neighborhoodFunction = GaussianFunction(radius=5)
som = SelfOrganizingMap(matrix_height=15,
                        matrix_width=15,
                        input_length=2,
                        neighborhood_function=neighborhoodFunction,
                        learning_rate=0.8,
                        minimum_tiredness_potential=0.8)

# data = data_import.generate_heart()
data = data_import.read_file('attract.txt')

data = data_import.scale_data_set_to_range(data, 0, 0.8)

printer.print_neurons_connections_over_data_points(som, data, width=900, height=900)

som.learn(data, 50)

printer.print_neurons_connections_over_data_points(som, data, width=900, height=900)
