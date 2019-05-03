import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import linalg
from draw_epipolar import plot_epipolar_line
from ransac import ransac


def read_image(filename1):
    img = cv2.imread(filename1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def feature_match(img1, img2):
    sift_detector = cv2.xfeatures2d.SIFT_create()
    # sift_detector = cv2.AKAZE_create()

    keypoints1, descriptor1 = sift_detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift_detector.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    # sort the matches in the order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=2)
    plt.imshow(img3)
    plt.show()

    matching_points1 = []
    matching_points2 = []
    for pair in matches:
        matching_points1.append(keypoints1[pair.queryIdx].pt)
        matching_points2.append(keypoints2[pair.trainIdx].pt)

    return np.array(matching_points1), np.array(matching_points2), matches


def calculate_fundamental_mat_ransac(data, model, maxiter, match_theshold):
    return ransac(data, model, 8, maxiter, match_theshold, int(0.5 * data.shape[0]), return_all=True)


class EightPointModel(object):
    """8 point algorithm model support the interface needed by ransac
    """

    def fit(self, data):
        """
        compute the fundamental matrix from corresponding points (x1, x2) using 8 point algorithm.
        """
        x1 = data[:, 0:2]
        x2 = data[:, 2:4]
        n_points = x1.shape[0]
        # build matrix for equations
        A = np.zeros((n_points, 9))
        for i in range(n_points):
            A[i] = [x1[i, 0] * x2[i, 0],
                    x1[i, 0] * x2[i, 1],
                    x1[i, 0],
                    x1[i, 1] * x2[i, 0],
                    x1[i, 1] * x2[i, 1],
                    x1[i, 1],
                    x2[i, 0],
                    x2[i, 1],
                    1]

        # compute linear least square solution
        U, S, V = linalg.svd(A)
        F = V[-1].reshape(3, 3)

        # constrain F
        # make rank 2 by zeroing out last singular value
        U, S, V = linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))

        return F / F[2, 2]

    def get_error(self, data, F):
        # data shape: [num_point, 4]
        x1 = data[:, 0:2]
        x2 = data[:, 2:4]
        # Sampson distance as error measure
        ones = np.ones([x1.shape[0], 1])
        x1 = np.hstack([x1, ones]).transpose()
        x2 = np.hstack([x2, ones]).transpose()
        Fx1 = np.dot(F, x1)  # Fx1: [3, 3] * [3, num_points]
        Fx2 = np.dot(F, x2)  # Fx2: [3, 3] * [3, num_points]
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


if __name__ == '__main__':
    # stereo reconstruction for rectified image pairs
    # img0 = read_image('data/rectified/01/im0.png')
    # img1 = read_image('data/rectified/01/im1.png')
    # _, _, _ = feature_match(img0, img1)

    # stereo reconstruction for non-rectified image pairs
    img1 = read_image('data/non-rectified/01/im0.png')
    img2 = read_image('data/non-rectified/01/im1.png')
    matching_points1, matching_points2, matches = feature_match(img1, img2)
    data = np.hstack([matching_points1, matching_points2])
    # F, ransac_data = calculate_fundamental_mat_ransac(data, EightPointModel(), maxiter=500, match_theshold=1e-1)
    # print(F)
    F, mask = cv2.findFundamentalMat(matching_points1, matching_points2, cv2.FM_RANSAC)
    print(F)
    print("inliner num:", np.sum(mask == 1))
    points1 = matching_points1[mask.ravel() == 1]
    points2 = matching_points2[mask.ravel() == 1]

    # show epipolar line
    points1 = np.hstack([points1, np.ones([points1.shape[0], 1])])
    plt.imshow(img1)
    for i in range(5):
        plot_epipolar_line(img1, F, points1[i])
    plt.show()

    points2 = np.hstack([points2, np.ones([points2.shape[0], 1])])
    plt.imshow(img2)
    for i in range(5):
        plot_epipolar_line(img2, F.transpose(), points2[i])
    plt.show()

    # # Find epilines corresponding to points in right image (second image) and
    # # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    # lines1 = lines1.reshape(-1, 3)
    # img5, img6 = drawlines(img1, img2, lines1, points1, points2)
    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    # lines2 = lines2.reshape(-1, 3)
    # img3, img4 = drawlines(img2, img1, lines2, points2, points1)
    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.show()