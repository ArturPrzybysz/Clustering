import os

from SelfOrganizingMap import SelfOrganizingMap
from PIL import Image
import numpy as np
import jsonpickle
import scipy.misc
from config import ROOT_DIR
from scipy.misc import imsave


class Thing(object):
    def __init__(self, name):
        self.name = name


def compress_image(image_file_name: str, target_file_name: str, som: SelfOrganizingMap.SelfOrganizingMap):
    path = os.path.join(ROOT_DIR, "data", image_file_name)
    image = Image.open(path)

    data = np.array(image)

    img_vectors = image_to_vectors(image_file_name)
    image_dimensions = [len(data) - len(data) % 2, len(data[0]) - len(data[0]) % 2]

    dictionary = []
    mapped_image = []

    for vec in img_vectors:
        closest_neurons_weights = som.react_to_input(vec)
        if not is_neuron_in_dictionary(closest_neurons_weights, dictionary):
            dictionary.append(closest_neurons_weights.tolist())
        mapped_image.append(first_index_of_vector_in_array(closest_neurons_weights, dictionary))

    dictionary = (np.rint((np.array(dictionary) * 255))).tolist()

    x = [image_dimensions, dictionary, mapped_image]
    obj = Thing(x)
    frozen = jsonpickle.encode(obj)
    with open(os.path.join(ROOT_DIR, "compressed", target_file_name), "w") as text_file:
        print(frozen, file=text_file)


def first_index_of_vector_in_array(vector, array):
    for i in range(len(array)):
        if np.all(array[i] == vector):
            return i
    raise Exception("Vector not found in array.")


def is_neuron_in_dictionary(neuron: np.array, dictionary: [np.array]):
    for d in dictionary:
        if np.all(neuron == d):
            return True
    return False


def decompress_image(encoded_file_name, decoded_file_name: str):
    with open(os.path.join(ROOT_DIR, "compressed", encoded_file_name), "r") as f:
        file = f.read()
    data = jsonpickle.decode(file).name

    image_dimensions = data[0]
    dictionary = data[1]
    image_map = data[2]

    M = unpack(image_map, dictionary, image_dimensions)
    imsave(os.path.join(ROOT_DIR, "decompressed", decoded_file_name), M)


def unpack(image_map, dictionary, image_dimensions):
    width = image_dimensions[0]
    height = image_dimensions[1]

    substitution_matrix = []

    for i in image_map:
        substitution_matrix.append(dictionary[i])

    substitution_matrix = np.array(substitution_matrix)
    print(width, height)
    rgb_matrix = np.zeros(width * height * 3).reshape(width, height, 3)  # OK!!!

    for i in range(len(substitution_matrix)):
        for j in range(len(substitution_matrix[0])):
            y = i % (int(width / 2)) * 2  # OK!!!
            if j in range(6, 12):
                y += 1

            x = int((i * 2) / width) * 2
            if j in range(3, 6) or j in range(9, 12):
                x += 1

            z = j % 3

            # print(x, y, z)

            rgb_matrix[y][x][z] = substitution_matrix[i][j]
    return rgb_matrix


def prepare_SOM(image_file_name: str, som: SelfOrganizingMap.SelfOrganizingMap, epochs):
    data = image_to_vectors(image_file_name)
    som.learn(data, epochs, visualize=False)

    return som


def image_to_vectors(image_file_name: str):
    path = os.path.join(ROOT_DIR, "data", image_file_name)
    img = Image.open(path)

    data = np.array(img)

    height = len(data) - len(data) % 2
    width = len(data[0]) - len(data[0]) % 2
    data = data[:height, :width]
    data = split_matrix_to_vectors(data)
    data = data / 255
    return data


def split_matrix_to_vectors(matrix: np.array):
    vectors = []
    print(len(matrix), len(matrix[0]))
    for i in np.arange(0, len(matrix), 2):
        for j in np.arange(0, len(matrix[0]), 2):
            vec = [[matrix[i][j],
                    matrix[i + 1][j],
                    matrix[i][j + 1],
                    matrix[i + 1][j + 1]]]
            vec = np.array(vec).flatten()
            vectors.append(vec)
    vectors = np.array(vectors)

    return vectors
