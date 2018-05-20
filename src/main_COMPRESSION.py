import SelfOrganizingMap.NeighborhoodFunction.GaussianFunction as neighborhood
import SelfOrganizingMap.SelfOrganizingMap as som
import FileCompression.Compression as compression

neighborhoodFunction = neighborhood.GaussianFunction(radius=1)
network = som.SelfOrganizingMap(matrix_height=4,
                                matrix_width=4,
                                input_length=12,
                                neighborhood_function=neighborhoodFunction,
                                learning_rate=0.7,
                                minimum_tiredness_potential=0.75)

network = compression.prepare_SOM("a.bmp", network, epochs=1)
compression.compress_image(image_file_name="a.bmp",
                           som=network,
                           target_file_name="TEST123.art")
compression.decompress_image("TEST123.art", "abc.bmp")
