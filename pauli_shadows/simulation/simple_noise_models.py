import numpy as np


def bit_flip(sample, p):

    n = len(sample)
    r = np.random.random(size=n)

    for i in range(n):
        if r[i] < p:
            x = sample[i]
            sample[i] = (x + 1) % 2

    return


def depolarize_local(sample, p):

    n = len(sample)
    r = np.random.random(size=n)
    flip_probability = 2.0 * p / 3.0

    for i in range(n):
        if r[i] < flip_probability:
            x = sample[i]
            sample[i] = (x + 1) % 2

    return


def depolarize_global(sample, p):

    n = len(sample)
    r = np.random.random()
    scramble_probability = 1.0 / (4**n - 1) + 1.0
    scramble_probability *= p

    if r < scramble_probability:
        for i in range(n):
            x = np.random.randint(2)
            sample[i] = x

    return


def amplitude_damp(sample, p):

    n = len(sample)
    r = np.random.random(size=n)

    for i in range(n):
        if sample[i] == 1 and r[i] < p:
            sample[i] = 0

    return
