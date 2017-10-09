import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from threshold_utils import perform_sobel_magnitude_directional_thresholding
from transform_utils import image_warp

def perform_undistort_and_threshold(images, mtx, dist):
    for frame in images:
        # Read in each image
        image = mpimg.imread(frame)

        # Use calibrate_and_undistort helper function to obtain undistorted image
        undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
        # plt.imshow(undistorted_image)
        # plt.show()

        # transform image to top-down
        transformed_image = image_warp(undistorted_image)
        plt.imshow(transformed_image)
        plt.show()

        # gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
        # plt.imshow(gray)
        # plt.show()

        combined_threshold_image = perform_sobel_magnitude_directional_thresholding(undistorted_image)
        # plt.imshow(combined_threshold_image, cmap='gray')
        # plt.show()

        # transform threshold image to top-down
        transformed_threshold_image = image_warp(combined_threshold_image)
        plt.imshow(transformed_threshold_image)
        plt.show()




