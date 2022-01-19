import numpy as np

data = {}


def add_value(key, value):
    if key not in data:
        data[key] = []

    data[key].append(value)


def get_average(key):
    return np.mean(data[key])


def get_sum(key):
    return np.sum(data[key])


def get_count(key):
    return len(data[key])


def increment(key):
    if key not in data:
        data[key] = 0

    data[key] += 1


def get(key, default=None):
    return data[key] if key in data else default
