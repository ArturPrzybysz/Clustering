from src.SelfOrganizingMap.NeighborhoodFunction.NeighborhoodFunction import NeighborhoodFunction


class WinnerTakeAll(NeighborhoodFunction):

    def apply(self, self_organizing_map, winner_neuron, learning_rate):
        self_organizing_map.neurons[winner_neuron.x][winner_neuron.y] += learning_rate
