import cv2
import numpy as np
import random

def calculating_dist(d1, d2):
    a = d1.shape[0]
    b = d2.shape[0]
    dot1 = (d1 * d1).sum(axis=1).reshape((a, 1)) * np.ones(shape=(1, b))
    dot2 = (d2 * d2).sum(axis=1) * np.ones(shape=(a, 1))
    dot_square = dot1 + dot2 - (2 * d1.dot(d2.T))

    return dot_square


def solution(left_img, right_img):
    left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("left_gray.jpg", left)
    right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("right_gray.jpg", right)
    features = cv2.xfeatures2d.SIFT_create()
    k1, d1 = features.detectAndCompute(left, None)
    k2, d2 = features.detectAndCompute(right, None)

    distance = calculating_dist(d1, d2)
    coord_left = np.array([k1[p].pt for p in np.where(distance < 7600)[0]])
    coord_right = np.array([k2[p].pt for p in np.where(distance < 7600)[1]])
    matched12 = np.concatenate((coord_left, coord_right), axis=1)

    inlier_matched12 = random.randint(20, 30)
    homo12 = []
    for i in range(1000):
        r = random.SystemRandom()
        rand_matching = np.concatenate(
            ([r.choice(matched12)], [r.choice(matched12)], [r.choice(matched12)], [r.choice(matched12)]), axis=0)
        Homography_matrix = cv2.getPerspectiveTransform(np.float32(rand_matching[:, 0:2]),
                                                        np.float32(rand_matching[:, 2:4]))
        if (np.linalg.matrix_rank(Homography_matrix) < 3):
            continue
        p1 = np.concatenate((matched12[:, 0:2], np.ones((len(matched12), 1))), axis=1)
        p2 = matched12[:, 2:4]
        cp = np.zeros((len(matched12), 2))
        while (i < len(matched12)):
            transform_matrix = np.matmul(Homography_matrix, p1[i])
            cp[i] = (transform_matrix / transform_matrix[2])[0:2]
            i += 1
        inl = matched12[np.where((np.linalg.norm(p2 - cp, axis=1) ** 2) < 0.5)[0]]
        if (len(inl) > inlier_matched12):
            inlier_matched12 = len(inl)
            homo12 = Homography_matrix.copy()
    result12 = cv2.warpPerspective(left_img, homo12, (
    int(left_img.shape[1] + right_img.shape[1] * 0.8), int(left_img.shape[0] + right_img.shape[0] * 0.4)))
    result12[0:right_img.shape[0], 0:right_img.shape[1]] = right_img

    distance = calculating_dist(d2, d1)
    coord_left = np.array([k2[p].pt for p in np.where(distance < 7600)[0]])
    coord_right = np.array([k1[p].pt for p in np.where(distance < 7600)[1]])
    matched21 = np.concatenate((coord_left, coord_right), axis=1)

    inlier_matched21 = random.randint(20, 30)
    homo21 = []
    for i in range(1000):
        r = random.SystemRandom()
        rand_matching = np.concatenate(
            ([r.choice(matched21)], [r.choice(matched21)], [r.choice(matched21)], [r.choice(matched21)]), axis=0)
        Homography_matrix = cv2.getPerspectiveTransform(np.float32(rand_matching[:, 0:2]),
                                                        np.float32(rand_matching[:, 2:4]))
        if (np.linalg.matrix_rank(Homography_matrix) < 3):
            continue
        p1 = np.concatenate((matched21[:, 0:2], np.ones((len(matched21), 1))), axis=1)
        p2 = matched21[:, 2:4]
        cp = np.zeros((len(matched21), 2))
        while (i < len(matched21)):
            transform_matrix = np.matmul(Homography_matrix, p1[i])
            cp[i] = (transform_matrix / transform_matrix[2])[0:2]
            i += 1
        inl = matched21[np.where((np.linalg.norm(p2 - cp, axis=1) ** 2) < 0.5)[0]]
        if (len(inl) > inlier_matched21):
            inlier_matched21 = len(inl)
            homo21 = Homography_matrix.copy()
    result21 = cv2.warpPerspective(right_img, homo21, (
    int(right_img.shape[1] + left_img.shape[1] * 0.8), int(right_img.shape[0] + left_img.shape[0] * 0.4)))
    result21[0:left_img.shape[0], 0:left_img.shape[1]] = left_img

    b1 = 0
    for i in range(result12.shape[0]):
        for j in range(result12.shape[1]):
            if (np.array_equal(result12[i, j], np.zeros(3))):
                b1 += 1

    b2 = 0
    for i in range(result21.shape[0]):
        for j in range(result21.shape[1]):
            if (np.array_equal(result21[i, j], np.zeros(3))):
                b2 += 1

    if (b1 < b2):
        result_img = result12
    else:
        result_img = result21
    return result_img


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)