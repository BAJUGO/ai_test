import numpy as np

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