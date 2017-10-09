import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Four source coordinates from test images (approximation)
src = np.float32([[585, 450], [695, 450], [1125, 720], [200, 720]])

# Four desired coordinates on transformed (top-down) view
dst = np.float32([[300, 0], [1000, 0], [1000, 720], [300, 720]])

def image_warp(image):
    image_size = (image.shape[1], image.shape[0])
    # compute perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # compute inverse perspective transform
    # Minv = cv2.getPerspectiveTransform(dst, src)

    # warp the image
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped