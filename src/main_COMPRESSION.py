import SelfOrganizingMap.NeighborhoodFunction.GaussianFunction as neighborhood
import SelfOrganizingMap.SelfOrganizingMap as som
import FileCompression.Compression as compression

neighborhoodFunction = neighborhood.GaussianFunction(radius=2)
network = som.SelfOrganizingMap(matrix_height=8,
                                matrix_width=8,
                                input_length=12,
                                neighborhood_function=neighborhoodFunction,
                                learning_rate=0.7,
                                minimum_tiredness_potential=0.75)

network = compression.prepare_SOM("rudy.bmp", network, epochs=4)
compression.compress_image(image_file_name="rudy.bmp",
                           som=network,
                           target_file_name="TEST123.art")
compression.decompress_image("TEST123.art", "rudy.bmp")
