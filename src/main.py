import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from utils import perform_image_calibration_and_undistort

### Parameters ###
nx = 9
ny = 6

### Helper Functions ###




# Read in all calibration images
source_path = '../camera_cal/'
filename = 'calibration*.jpg' # images have similar name with *
images = glob.glob(source_path + filename) # glob API to read all images


obj_points, image_points = perform_image_calibration_and_undistort(images, nx, ny)




