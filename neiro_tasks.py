import numpy as np


def neiron_input(value, weight):
    return value*weight


def sigmoid(x):
    return 1/(1+np.exp(-x))


def delta_out_sigmoid(ideal, actual):
    return (ideal - actual) * (1 - actual) * actual


