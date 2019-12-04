import numpy as np
from scipy import ndimage


def rgb2gray(rgb):
    """
        Transform an RGB image to gray image.
    """
    return rgb[..., :3] @ [0.2989, 0.5870, 0.1140]


def converge(U, A, B, I, *, time=0, step=1):
    """
        Get CellNN's steady-state by using Euler's integration method.
    """
    steady_flag = False
    t = time
    X = U
    Y = (np.abs(X + 1) - np.abs(X - 1)) / 2
    while True:
        X1 = -X + ndimage.convolve(Y, A) + ndimage.convolve(U, B) + I
        X1 = X + step * X1
        t = t - 1
        Y1 = (np.abs(X1 + 1) - np.abs(X1 - 1)) / 2
        delta = np.abs(Y1 - Y)
        X = X1
        Y = Y1
        if ~delta.any():
            steady_flag = True
        if steady_flag:
            break
        elif (time != 0) and (t <= 0):
            break
    return Y
