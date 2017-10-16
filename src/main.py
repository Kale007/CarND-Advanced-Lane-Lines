import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from calib_utils import perform_image_calibration_and_undistort_chessboards, show_images
from undistort_and_thresh_utils import perform_undistort_and_threshold



# Read in all calibration images
chessboard_source_path = '../camera_cal/'
chessboard_filename = 'calibration1*.jpg' # images have similar name with *
chessboard_images = glob.glob(chessboard_source_path + chessboard_filename) # glob API to read all images


mtx, dst = perform_image_calibration_and_undistort_chessboards(chessboard_images)

test_images_source_path = '../test_images/'
test_images_filename = 'test*.jpg'



test_images = glob.glob(test_images_source_path + test_images_filename)
# show_images(test_images)
# show_undistorted_images(test_images, mtx, dst)

is_video = False
perform_undistort_and_threshold(test_images, mtx, dst, is_video, None)


# Perform on video
is_video = True
video_file = '../project_video.mp4'
perform_undistort_and_threshold(None, mtx, dst, is_video, video_file)
