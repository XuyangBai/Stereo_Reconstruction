import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import linalg
from draw_epipolar import plot_epipolar_line
from ransac import ransac


def read_image(filename1):
    img = cv2.imread(filename1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def feature_match(img1, img2):
    sift_detector = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptor1 = sift_detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift_detector.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    # sort the matches in the order of distance
    # good_matches = sorted(matches, key=lambda x: x.distance)

    good_matches = []
    # Apply ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:10], None, flags=2)
    # plt.imshow(img3)
    # plt.show()

    matching_points1 = []
    matching_points2 = []
    for pair in good_matches:
        matching_points1.append(keypoints1[pair.queryIdx].pt)
        matching_points2.append(keypoints2[pair.trainIdx].pt)

    print('matching num:', len(good_matches))
    return np.array(matching_points1), np.array(matching_points2), good_matches


def calculate_fundamental_mat_ransac(data, model, maxiter, match_theshold):
    F, ransac_data = ransac(data, model, 8, maxiter, match_theshold, int(0.5 * data.shape[0]), return_all=True)
    mask = np.zeros([data.shape[0], 1])
    mask[ransac_data['inliers']] = 1
    return F, mask


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

        # constrain F: make rank 2 by zeroing out last singular value
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
        Fx1 = np.dot(F, x1)  # Fx1: [3, 3] * [3, num_points] = [3, num_points]
        Fx2 = np.dot(F, x2)  # Fx2: [3, 3] * [3, num_points] = [3, num_points]
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


def decompose_essential_mat(E, pts1, pts2):
    def in_front_of_both_cameras(first_points, second_points, rot, trans):
        # check if the point correspondences are in front of both images
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :], second)
            first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True

    # decompose Essential matrix into R, t
    U, S, V = linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    first_inliers = np.hstack([pts1, np.ones([pts1.shape[0], 1])])
    second_inliers = np.hstack([pts2, np.ones([pts1.shape[0], 1])])

    # Determine the correct choice of second camera matrix
    # only in one of the four configurations will all the points be in front of both cameras
    # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
    R = U.dot(W).dot(V)
    T = U[:, 2]
    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

        # Second choice: R = U * W * Vt, T = -u_3
        T = - U[:, 2]
        if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(V)
            T = U[:, 2]

            if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]
    return R, T


def show_epipolar(img1, img2, pts1, pts2, F, filename, show_epipole):
    # show epipolar line
    plt.figure()
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    points1_homo = np.hstack([pts1, np.ones([pts1.shape[0], 1])])
    points2_homo = np.hstack([pts2, np.ones([pts2.shape[0], 1])])
    plt.subplot(121)
    plt.imshow(img1)
    for i in range(5):
        # draw the epipolar line for points2[i] on img1
        plot_epipolar_line(img1, F, points2_homo[i], show_epipole=show_epipole)

    plt.subplot(122)
    plt.imshow(img2)
    for i in range(5):
        # draw the epipolar line for points2[i] on img2
        plot_epipolar_line(img2, F.transpose(), points1_homo[i], show_epipole=show_epipole)
    # plt.show()
    plt.savefig("./data/epipolar_line/{}.png".format(filename))


def draw_epipolar(img1, img2, filename):
    # find the corrsponding points using SIFT feature.
    matching_points1, matching_points2, matches = feature_match(img1, img2)
    data = np.hstack([matching_points1, matching_points2])
    # estimate the fundamental matrix using RANSAC based 8 point algorithm
    # F, mask = calculate_fundamental_mat_ransac(data, EightPointModel(), maxiter=10000, match_theshold=0.1)
    # print(F)
    # print("inliner num:", np.sum(mask == 1))
    F, mask = cv2.findFundamentalMat(matching_points1, matching_points2, cv2.FM_RANSAC)
    print(F)
    print("inliner num:", np.sum(mask == 1))
    # select only inlier points
    pts1 = matching_points1[mask.ravel() == 1]
    pts2 = matching_points2[mask.ravel() == 1]

    # show_epipolar(img1, img2, pts1, pts2, F, show_epipole=True)
    show_epipolar(img1, img2, pts1, pts2, F, filename=filename, show_epipole=False)


if __name__ == '__main__':
    for filename in os.listdir("./data/non-rectified"):
        # stereo reconstruction for non-rectified image pairs
        if filename != 'cameras.txt':
            img1 = read_image('data/non-rectified/{0}/im0.png'.format(filename))
            img2 = read_image('data/non-rectified/{0}/im1.png'.format(filename))
            draw_epipolar(img1, img2, filename)

    # K1 = np.array([
    #     [541.911, 0, 499.646],
    #     [0, 541.618, 231.773],
    #     [0, 0, 1],
    # ])
    # K2 = np.array([
    #     [538.731, 0, 503.622],
    #     [0, 538.615, 265.447],
    #     [0, 0, 1],
    # ])
    # K1_inv = linalg.inv(K1)
    # K2_inv = linalg.inv(K2)
    # get essential matrix from F
    # E1 = np.dot(K1, np.dot(F, K1_inv))
    # E2 = np.dot(K1, np.dot(F, K2_inv))
    # #
    # # # decompose essential matrix E to get R and t
    # R1, t1 = decompose_essential_mat(E1, pts1, pts2)
    # R2, t2 = decompose_essential_mat(E2, pts1, pts2)
    # print("R1", R1)
    # print("R2", R2)
    # # print(R1, t1)
    # # print(R2, t2)
    # # print(linalg.inv(R1).dot(R2))
    # # print(t2 - t1)
    # # perform the rectification
    # d = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, 8)
    # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d, K2, d, img1.shape[:2], linalg.inv(R1).dot(R2), t1 - t2, alpha=1.0)
    # print("R1", R1)
    # print("R2", R2)
    #
    # # R1 = np.eye(3)
    # # R2 = np.eye(3)
    # mapx1, mapy1 = cv2.initUndistortRectifyMap(K1, d, R1, K1, img1.shape[:2], cv2.CV_32F)
    # mapx2, mapy2 = cv2.initUndistortRectifyMap(K2, d, R2, K2, img2.shape[:2], cv2.CV_32F)
    # img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
    # img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
    # plt.subplot(121)
    # plt.imshow(img_rect1)
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(img_rect2)
    # plt.show()
    # # plt.show()
    #
    # # draw the images side by side
    # # total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1])
    # # img = np.zeros(total_size, dtype=np.uint8)
    # # img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
    # # img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
    # #
    # # draw horizontal lines every 25 px accross the side by side image
    # # for i in range(20, img.shape[0], 25):
    # #     cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
    #
    # # cv2.imshow('rectified', img)
    # # cv2.waitKey(0)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
