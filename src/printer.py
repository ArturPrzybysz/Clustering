from PIL import Image, ImageDraw
import os


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
    im = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(im)

    # dot_size = 2
    # _draw_points(data, draw, dot_size)

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


def _draw_points(data, draw, dot_size):
    for d in data:
        _draw_point(d[0] * 700 + 100, d[1] * 700 + 100, dot_size, draw)


def _draw_point(x, y, size, draw):
    x1 = int(x - size / 2)
    y1 = int(y - size / 2)
    x2 = x1 + size
    y2 = y1 + size
    draw.ellipse(xy=([(x1, y1), (x2, y2)]), fill=(0, 0, 0))


def _save_photo(draw, name):
    pass
