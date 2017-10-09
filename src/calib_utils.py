import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def perform_image_calibration_and_undistort_chessboards(chessboard_images):
    # Chessboard parameters
    nx = 9
    ny = 6
    image_shape = mpimg.imread(chessboard_images[0]).shape[1::-1] # All images have same shape

    objpoints, imgpoints = get_objpoints_and_imgpoints(chessboard_images, nx, ny)

    # Calibrate camera using objpoints, imgpoints
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    return mtx, dist

# calculates objpoints and imgpoints for camera calibration
def get_objpoints_and_imgpoints(images, nx, ny):
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane


    # Prepare object points, like (0, 0, 0), (1, 0, 0), etc
    objp = np.zeros((nx * ny ,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2) # x and y coordinates

    for file in images:
        # Read in each image
        image = mpimg.imread(file)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display corners
            image = cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
            # plt.imshow(image)
            # plt.show()

    return objpoints, imgpoints


def show_images(images):
    for file in images:
        image = mpimg.imread(file)
        plt.imshow(image)
        plt.show()