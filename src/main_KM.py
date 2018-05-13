import data_import
import numpy as np

from KMeans.KMeans import k_means

dataA = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.2, 0.4)
dataB = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.4, 0.5)
dataC = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.6, 0.75)
dataD = data_import.scale_data_set_to_range(data_import.generate_heart(), 0.8, 1)

data = np.concatenate((dataA, dataB, dataC, dataD))

clusters = k_means(data, cluster_count=4, epochs=11, dimensions=2)
