import numpy as np


def vrow(array):
    """
    shape 1D np array to row array
    """
    return np.reshape(array, (1, np.size(array)))


def vcol(array):
    """
    shape 1D np array to columnt array
    """
    return np.reshape(array, (np.size(array), 1))
