import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]


def decode_inst(value):
    if value[0] == 1:
        return 0
    elif value[1] == 1:
        return 2
    elif value[2] == 1:
        return 5
    elif value[3] == 1:
        return 3
    else:
        return 4


def encode(value):
    # what is the meaning of those controls?
    # 1 of k encoding
    if value == 2.0:
        return [1, 0, 0, 0]
    elif value == 5.0:
        return [0, 1, 0, 0]
    elif value == 3.0:
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]


def encode4(value):
    if value == 5.0:
        return [0, 1, 0, 0]
    elif value == 3.0 or value == 6.0:
        return [0, 0, 1, 0]
    elif value == 4.0 or value == 7.0 or value == 8.0:
        return [0, 0, 0, 1]
    else:
        return [1, 0, 0, 0]


def encode8(value):
    array = [0] * 8
    array[value] = 1
    return array


def decode(vector):
    dec_vec = []

    for i in range(vector.shape[0]):
        dec_vec.append(decode_inst(vector[i, :]))

    return np.array(dec_vec)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_distance(value):
    if not is_number(value):
        return 300.0  # A big enought number to not consider the distance
    if value < 0.0:
        return 300.0  # The distance cannot be negative... this is really a problem.
    return value
