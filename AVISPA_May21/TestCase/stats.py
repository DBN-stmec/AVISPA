import numpy as np

logged_values = {}
limits = {}
count_limits = {}


def set(key, value):
    logged_values[key] = value

    check_limits(key)


def reset():
    global logged_values
    logged_values = {}


def append(key, value):
    if key in logged_values.keys():
        logged_values[key].append(value)
    else:
        logged_values[key] = [value]

    check_limits(key)


def check_limits(key):
    if key in count_limits:
        if count_limits[key] < len(logged_values[key]):
            logged_values[key].pop(0)

    if key in limits:
        if logged_values[key] >= limits[key]:
            logged_values[key] = limits[key]


def limit_count(key, limit):
    count_limits[key] = limit


def limit(key, limit):
    limits[key] = limit


def increment(key):
    if key in logged_values.keys():
        logged_values[key] += 1
    else:
        logged_values[key] = 1

    check_limits(key)


def get(key, default=None):
    if key in logged_values:
        return logged_values[key]

    return default


def get_sum(key, default=0):
    return np.sum(get(key, default))


def get_average(key, default=0):
    return np.average(get(key, default))


def get_length(key):
    return len(get(key))


def get_last(key, limit=1):
    values = get(key, [])
    return values[-limit:] if len(values) else []