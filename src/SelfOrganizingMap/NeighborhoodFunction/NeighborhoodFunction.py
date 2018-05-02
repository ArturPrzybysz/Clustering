from abc import ABC, abstractmethod

from src.SelfOrganizingMap.Neuron import Neuron
from src.SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap


class NeighborhoodFunction(ABC):
    @abstractmethod
    def apply(self, self_organizing_map: SelfOrganizingMap, winner_neuron: Neuron, learning_rate: float):
        pass
