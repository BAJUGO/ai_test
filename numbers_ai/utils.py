import numpy as np
from matplotlib import pyplot as plt


def load_dataset():
    with np.load("mnist.npz") as f:
        x_train = f['x_train'].astype('float32') / 255

        x_train = np.reshape(x_train, (60000, 784))

        y_train = f['y_train']

        y_train = np.eye(10)[y_train]
        # y_train изначально - какой-то массив чисел. После чего
        # мы создаём единичную матрицу 10 на 10, и берём из неё элементы от 0 до 9, ибо сам
        # y_train это массив с такими числами.

        return x_train, y_train


def sigmoid(x):
    return 1/(1+np.exp(-x))


def change_bias_inp(learning_rate, delta_hidden):
    return learning_rate * delta_hidden

def change_bias_hid(learning_rate, delta_output):
    return learning_rate * delta_output

def change_weights_inp_to_hid(learning_rate, delta_hidden, image):
    return learning_rate * delta_hidden @ np.transpose(image)

def change_weights_hid_to_inp(learning_rate, delta_output, hidden):
    return learning_rate * delta_output @ np.transpose(hidden)


def load_twenty_epochs():
    with np.load("20_epoch.npz") as f:
        weights_inp_to_hid = f["w1"]
        weights_hid_to_out = f["w2"]
        bias_inp_to_hid = f["b1"]
        bias_hid_to_out = f["b2"]
        return weights_inp_to_hid, weights_hid_to_out, bias_inp_to_hid, bias_hid_to_out