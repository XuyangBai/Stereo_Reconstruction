import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import cv2


def compute_epipole(F):
    """ Computes the (right) epipole from a fundamental matrix F.
        (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(img, F, x, epipole=None, show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image."""
    m, n = img.shape[:2]
    line = np.dot(F, x)

    # epipolar line parameter and values
    t = np.linspace(0, n - 1, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')
        # plt.show()
