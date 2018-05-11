from abc import ABC, abstractmethod
from src.SelfOrganizingMap import SelfOrganizingMap
from SelfOrganizingMap import Neuron
import numpy as np


class NeighborhoodFunction(ABC):
    @abstractmethod
    def apply(self, som: SelfOrganizingMap, winner_neuron: Neuron, learning_rate: float, learning_point: np.array):
        pass
