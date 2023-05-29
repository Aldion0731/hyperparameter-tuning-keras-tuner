from dataclasses import dataclass

import numpy as np
from tensorflow import keras


@dataclass
class MnistData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def preprocess(self):
        self.X_train = self.x_train / 255.0
        self.X_test = self.X_test / 255.0


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return MnistData(x_train, x_test, y_train, y_test)
