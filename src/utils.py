import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def perform_image_calibration_and_undistort(images, nx, ny):
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane


  # Prepare object points, like (0, 0, 0), (1, 0, 0), etc
  objp = np.zeros((nx * ny ,3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2) # x and y coordinates
  
  for frame in images:
      # Read in each image
      image = mpimg.imread(frame)

      # Convert image to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Find chessboard corners
      ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

      # If corners are found, add object points, image points
      if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display corners
        image = cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
        plt.imshow(image)
        plt.show()

        # Use calibrate_and_undistort helper function
        undistorted = calibrate_and_undistor(image, objpoints, imgpoints)
        plt.imshow(undistorted)
        plt.show()

        return objpoints, imgpoints



# performs camera calibration and image undistortion, returns undistorted image
def calibrate_and_undistor(image, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

# performs perspective transform
def unwarp_corners(image, nx, ny, mtx, dst):
  print(np.shape(image))