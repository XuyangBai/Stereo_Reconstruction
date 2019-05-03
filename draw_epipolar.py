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
    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')
        # plt.show()

# def drawlines(img1, img2, lines, pts1, pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r, c = img1.shape
#     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#     for r, pt1, pt2 in zip(lines, pts1, pts2):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
#         img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
#         img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
#     return img1, img2
