from PIL import Image, ImageDraw
import os
import random

from NeuralGas import Neuron


def save_clusters_and_centroids(clusters, centroids, width, height, filename):
    im = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 0, 125)]
    if len(centroids) > len(colors):
        for i in range(len(centroids), len(colors)):
            colors.append(_random_color())

    for i in range(0, len(clusters)):
        _draw_points(clusters[i], draw, dot_size=2, color=colors[i])

    _draw_points(centroids, draw, dot_size=10, color=(0, 0, 0))
    path = os.path.dirname(os.path.abspath(__file__)) + "\\img\\" + filename
    im.save(path + ".png")


def save_neurons_over_data_points(neurons: [Neuron], data, height, width, filename):
    im = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(im)
    _draw_points(data, draw, dot_size=2)
    _draw_neurons(neurons, draw, dot_size=7)

    path = os.path.dirname(os.path.abspath(__file__)) + "\\img\\" + filename
    im.save(path + ".png")


def print_neurons_over_data_points(neurons: [Neuron], data, height, width):
    im = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(im)
    _draw_points(data, draw, dot_size=2)
    _draw_neurons(neurons, draw, dot_size=10)

    im.show()


def _random_color():
    return [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]


def _draw_neurons(neurons: [Neuron], draw, dot_size):
    for n in neurons:
        _draw_point(n.weights[0] * 700 + 100, n.weights[1] * 700 + 100, dot_size, draw, color=(255, 0, 0))


def save_neurons_connections_over_data_points(som, data, filename):
    im = Image.new('RGB', (900, 900), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    dot_size = 2
    _draw_points(data, draw, dot_size)

    connections = _prepare_connections(som)
    _draw_connections(connections, draw)
    path = os.path.dirname(os.path.abspath(__file__)) + "\\img\\" + filename
    im.save(path + ".png")


def print_neurons_connections_over_data_points(som, data, width: int, height: int):
    im = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    dot_size = 2
    _draw_points(data, draw, dot_size)

    connections = _prepare_connections(som)
    _draw_connections(connections, draw)

    im.show()


def _draw_connections(connections, draw):
    for c in connections:
        x1 = int(c[0][0] * 700) + 100
        x2 = int(c[1][0] * 700) + 100
        y1 = int(c[0][1] * 700) + 100
        y2 = int(c[1][1] * 700) + 100
        draw.line(([x1, y1, x2, y2]), fill=(255, 0, 0), width=2)


def _prepare_connections(som):
    connections = []
    for x in range(0, som.matrix_height - 1):
        for y in range(0, som.matrix_width):
            connections.append([som.neurons[x][y].weights, som.neurons[x + 1][y].weights])
    for y in range(0, som.matrix_width - 1):
        for x in range(0, som.matrix_height):
            connections.append([som.neurons[x][y].weights, som.neurons[x][y + 1].weights])
    return connections


def _draw_points(data, draw, dot_size, color=(0, 0, 0)):
    for d in data:
        _draw_point(d[0] * 700 + 100, d[1] * 700 + 100, dot_size, draw, color)


def _draw_point(x, y, size, draw, color):
    x1 = int(x - size / 2)
    y1 = int(y - size / 2)
    x2 = x1 + size
    y2 = y1 + size
    draw.ellipse(xy=([(x1, y1), (x2, y2)]), fill=color)
