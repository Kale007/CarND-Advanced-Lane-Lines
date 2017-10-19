import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from calib_utils import perform_image_calibration_and_undistort_chessboards, show_images
from pipeline import pipeline



### Read in all calibration images
chessboard_source_path = '../camera_cal/'
chessboard_filename = 'calibration*.jpg' # images have similar name with *
chessboard_images = glob.glob(chessboard_source_path + chessboard_filename) # glob API to read all images


mtx, dist = perform_image_calibration_and_undistort_chessboards(chessboard_images)

# image = mpimg.imread(chessboard_images[0])
# undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
# cv2.imwrite('../output_images/undistorted_chessboard.png', image)


### Read in test images
test_images_source_path = '../test_images/'
test_images_filename = 'test*.jpg'

test_images = glob.glob(test_images_source_path + test_images_filename)
# show_images(test_images)

is_video = False
pipeline(test_images, mtx, dist, is_video, None)



### Perform on video
is_video = True
video_file = '../project_video.mp4'
pipeline(None, mtx, dist, is_video, video_file)
