import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
cap = cv2.VideoCapture(0)

from threshold_utils import perform_color_and_gradient_thresholding
from transform_utils import image_warp
from detect_lanes_utils import perform_detect_lanes

def perform_undistort_and_threshold(images, mtx, dist, is_video, video_file):
    # perform image steps if input is not a video
    if is_video == False:
        # Indicate this is first frame in video for lane detection
        is_first_frame = True

        left_fit = []
        right_fit = []
        for frame in images:
            # Read in each image
            image = mpimg.imread(frame)

            # Use calibrate_and_undistort helper function to obtain undistorted image
            undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
            # plt.imshow(undistorted_image)
            # plt.show()

            # Transform image to top-down
            transformed_image, Minv = image_warp(undistorted_image)
            # plt.imshow(transformed_image)
            # plt.show()

            # gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
            # plt.imshow(gray)
            # plt.show()

            combined_threshold_image = perform_color_and_gradient_thresholding(undistorted_image)
            # plt.imshow(combined_threshold_image, cmap='gray')
            # plt.show()

            # Transform threshold image to top-down
            transformed_threshold_image, MinV = image_warp(combined_threshold_image)
            # plt.imshow(transformed_threshold_image)
            # plt.show()

            # Use sliding window search to detect lane lines
            left_fit, right_fit = perform_detect_lanes(transformed_threshold_image, is_first_frame, left_fit, right_fit, Minv, undistorted_image)

    if is_video:
        is_first_frame = True

        left_fit = []
        right_fit = []

        cap = cv2.VideoCapture(video_file)
        while(cap.isOpened()):
            ret, frame = cap.read()
            image = frame
            undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
            transformed_image, Minv = image_warp(undistorted_image)
            combined_threshold_image = perform_color_and_gradient_thresholding(undistorted_image)
            transformed_threshold_image, MinV = image_warp(combined_threshold_image)
            left_fit, right_fit = perform_detect_lanes(transformed_threshold_image, is_first_frame, left_fit, right_fit, Minv, undistorted_image)
            is_first_frame = False

            # Display the resulting frame
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()