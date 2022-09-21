# Panaroma-construction-using-image-stitching

The goal of this project is to stitch two images (named \left.jpg" and \right.jpg") together to construct a panorama image. To complete the task, we follow the following steps:

(1) Find keypoints (points of interest) in the given images, e.g., Harris detector or SIFT point detector.
(2) Use SIFT or other feature descriptors to extract features for these keypoints.
(3) Match the keypoints between two images by comparing their feature distance using KNN, k=2.
(4) Compute the homography matrix using RANSAC algorithm. 
(5) Use the homography matrix to stitch the two given images into a single panorama.

# Note: Use opencv == 3.4.2.17
