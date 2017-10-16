import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### Parameters ###

# color parameters
red_thresh = (200, 255)
green_thresh = (200, 255)
saturation_thresh = (170, 255)
lightness_thresh = (90, 255)

# gradient parameters
sobel_kernel = 3
sobel_thresh_x = (20, 100)
sobel_thresh_y = (20, 100)
mag_thresh = (30, 100)
dir_thresh = (0.7, 1.3)




def perform_color_and_gradient_thresholding(undisorted_image):
    combined_color_threshold = perform_color_thresholding(undisorted_image)
    combined_gradient_threshold = perform_sobel_magnitude_directional_thresholding(undisorted_image)

    combined_color_and_gradient_threshold = np.zeros_like(combined_color_threshold)
    combined_color_and_gradient_threshold[(combined_color_threshold == 1) | (combined_gradient_threshold == 1)] = 1

    return combined_color_and_gradient_threshold

def perform_color_thresholding(image):
    red = red_threshold(image, red_thresh)
    green = green_threshold(image, green_thresh)
    saturation = saturation_threshold(image, saturation_thresh)
    lightness = lightness_threshold(image, lightness_thresh)

    combined = np.zeros_like(red)
    # combined[((red == 1) & (green == 1)) | ((saturation == 1) & (lightness == 1))] = 1
    combined[(red == 1) & (saturation == 1)] = 1

    return combined

def perform_sobel_magnitude_directional_thresholding(image):
    ksize = sobel_kernel
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(gray, sobel_kernel=ksize, sobel_thresh=sobel_thresh_x, orient='x')
    grady = abs_sobel_threshold(gray, sobel_kernel=ksize, sobel_thresh=sobel_thresh_y, orient='y')
    mag_binary = mag_threshold(gray, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, dir_thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


# Color threshold helper functions
def red_threshold(image, red_thresh):
    red_channel = image[:,:,0]
    binary_red = np.zeros_like(red_channel)
    binary_red[(red_channel >= red_thresh[0]) & (red_channel <= red_thresh[1])] = 1
    return binary_red

def green_threshold(image, green_thresh):
    green_channel = image[:,:,0]
    binary_green = np.zeros_like(green_channel)
    binary_green[(green_channel >= green_thresh[0]) & (green_channel <= green_thresh[1])] = 1
    return binary_green

def saturation_threshold(image, s_thresh):
    # convert to hls colorspace
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    saturation = hls[:,:,2]
    binary_saturation = np.zeros_like(saturation)
    binary_saturation[(saturation >= saturation_thresh[0]) & (saturation <= saturation_thresh[1])] = 1
    return binary_saturation

def lightness_threshold(image, l_thresh):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    lightness = hls[:,:,1]
    binary_lightness = np.zeros_like(lightness)
    binary_lightness[(lightness >= lightness_thresh[0]) & (lightness <= lightness_thresh[1])] = 1
    return binary_lightness


# Gradient threshold helper functions
def abs_sobel_threshold(image, sobel_kernel, sobel_thresh, orient='x'):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    return grad_binary

def mag_threshold(image, sobel_kernel, mag_thresh):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel, dir_thresh):
    # Calculate gradient direction
    abs_sobelx = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_direction = np.arctan2(abs_sobelx, abs_sobely)
    dir_binary = np.zeros_like(grad_direction)
    dir_binary[(grad_direction >= dir_thresh[0]) & (grad_direction <= dir_thresh[1])] = 1
    return dir_binary



