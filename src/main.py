import random

from src.LearningData import LearningData
from src.SelfOrganizingMap.NeighborhoodFunction.GaussianFunction import GaussianFunction
from src.SelfOrganizingMap.SelfOrganizingMap import SelfOrganizingMap

neighborhoodFunction = GaussianFunction(3)
som = SelfOrganizingMap(15, 15, 2, neighborhoodFunction)

sample = [[random.random(), random.random()] for x in range(0, 1000)]

data = LearningData(sample)

som.learn(data, 150)
