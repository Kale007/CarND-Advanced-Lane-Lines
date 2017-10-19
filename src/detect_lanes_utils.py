import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def perform_detect_lanes(image, is_first_frame, left_fit, right_fit, Minv, undistorted_image):
    reg_of_interest_image, projected_image, left_fit, right_fit, left_curverad, right_curverad, distance_from_center = sliding_window_search(image, is_first_frame, left_fit, right_fit, Minv, undistorted_image)

    return reg_of_interest_image, projected_image, left_fit, right_fit, left_curverad, right_curverad, distance_from_center

def sliding_window_search(image, is_first_frame, left_fit, right_fit, Minv, undistorted_image):
    num_windows = 9
    margin = 100 # width of window +/- margin
    minpix = 50 # min number of pixels to recenter window


    out_img, leftx_base, rightx_base = find_image_histogram_peaks(image)
    window_height = np.int(image.shape[0]/num_windows)

    # If the frame is the first frame of video, perform

    # Get nonzero x and y positions
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    if is_first_frame:
        left_lane_inds = []
        right_lane_inds = []

        for window in range(num_windows):
            # identify window boundaried
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw windows
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)

            # Identify nonzero pixels
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If more than minpix are found, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, rightx = generate_polynomial(image, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    # If the frame is not the first frame of video
    else:
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, rightx = generate_polynomial(image, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    projected_image = project_onto_road(image, left_fitx, right_fitx, ploty, Minv, undistorted_image)


    ### project area


    # Visualize windows for lane detection
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()



    # Visualize region of interest to detect lanes
    reg_of_interest_image = visualize_region_of_interest(image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, margin)

    # Radius of curvature
    left_curverad, right_curverad, distance_from_center = calculate_radius_of_curvature(ploty, left_fitx, right_fitx)

    # Return vales to be used for next frame measurement (applicable for videos)
    return reg_of_interest_image, projected_image, left_fit, right_fit, left_curverad, right_curverad, distance_from_center

def find_image_histogram_peaks(image):
    # Take histogram of bottom half of image
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)

    # Creatae output image
    out_img = np.dstack((image, image, image))*255

    # Find peaks of left and right halves of histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return out_img, leftx_base, rightx_base

def generate_polynomial(image, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, rightx

def visualize_region_of_interest(image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, margin):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((image, image, image))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area and recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()

    return result


def calculate_radius_of_curvature(ploty, left_fitx, right_fitx):

    # measure curvature from bottom of image
    y_eval = np.max(ploty)

    left_fitx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    right_fitx = right_fitx[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension (30m distance / 720 pixels in y-direction)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension (3.7m lane width / 700 pixels between lanes in image)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    # Calculate width of road and car position relative to center of lane
    left_lane_pos = left_fit_cr[len(left_fit_cr)-1]
    right_lane_pos = right_fit_cr[len(right_fit_cr)-1]
    road_width = right_lane_pos - left_lane_pos
    center_of_lane = left_lane_pos + (road_width/2)
    position_of_car = 640 * xm_per_pix # center pixel * pixel-conversion (in meters)

    distance_from_center = position_of_car - center_of_lane

    return left_curverad, right_curverad, distance_from_center


def project_onto_road(image, left_fitx, right_fitx, ploty, Minv, undistorted_image):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the image blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    # cv2.imshow('result', result)

    return result









### convolutional sliding window search:
# # window settings parameters
# window_width = 50
# window_height = 120 # Breaks image into 720/120 = 6 veritcal layers
# margin = 50 # How much to slide left and right

# def sliding_window_search(image):
#     image = image.astype(np.uint8)
#     window_centroids = find_window_centroids(image, window_width, window_height, margin)


#     # If window centers were found
#     if len(window_centroids) > 0:

#         # Points used to draw all left and right windows
#         l_points, r_points = np.zeros_like(image), np.zeros_like(image)

#         # Draw windows at each level
#         for level in range(0, len(window_centroids)):
#             # Use window_mask function to draw window areas
#             l_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
#             r_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
#             # Add graphic points from window mask to total pixels found
#             l_points[(l_points == 255) | ((l_mask == 1))] = 255
#             r_points[(r_points == 255) | ((r_mask == 1))] = 255

#         # Draw the results
#         template = np.array(r_points+l_points, np.uint8) # add both left and right winow pixels together
#         zero_channel = np.zeros_like(template) # create a zero color channel
#         template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
#         warpage = np.dstack((image, image, image))*255 # making the original road pixel 3 color channels
#         output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0,) # overlay the original road image with window results

#     # If no window centers are found, just display original road map
#     else:
#         output = np.array(cv2.merge((image, image, image)), np.uint8)

#     # Fit polynomial
#     y = range(0, image.shape[0], window_height)

#     left_fit = np.polyfit([x[0] for x in window_centroids], y, 2)
#     right_fit = np.polyfit([x[1] for x in window_centroids], y, 2)

#     ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
#     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#     out_img = np.dstack((image, image, image))*255
#     out_img[y, output[0]] = [255, 0, 0]
#     out_img[y, output[1]] = [0, 0, 255]

#     plt.imshow(out_img)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)

#     return output

# def window_mask(width, height, image_ref, center, level):
#     output = np.zeros_like(image_ref)
#     output[int(image_ref.shape[0]-(level+1)*height):int(image_ref.shape[0]-level*height), max(0, int(center-width/2)):min(int(center+width/2), image_ref.shape[1])] = 1
#     return output

# def find_window_centroids(image, window_width, window_height, margin):

#     window_centroids = [] # store (left, right) window centroids at each level
#     window = np.ones(window_width) # window template

#     # Sum quarter bottom of image to get slice
#     l_sum = np.sum(image[int(3*image.shape[0]/4):, :int(image.shape[1]/2)], axis=0)
#     l_center = np.argmax(np.convolve(window, l_sum))-window_width/2
#     r_sum = np.sum(image[int(3*image.shape[0]/4):, int(image.shape[1]/2):], axis=0)
#     r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(image.shape[1]/2)

#     # Append l_center, r_center to first layer
#     window_centroids.append((l_center, r_center))

#     # Go through each layer and find max pixel location
#     for level in range(1, int(image.shape[0]/window_height)):
#         # Convolve window into vertical slice of image
#         image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
#         conv_signal = np.convolve(window, image_layer)

#         # Find best left centroid using previous l_center as reference
#         offset = window_width/2 #convolutional signal reference is at right side of window, not center
#         l_min_index = int(max(l_center+ offset-margin, 0))
#         l_max_index = int(min(l_center+offset+margin, image.shape[1]))
#         l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

#         # Find best right centroid using previous r_center as reference
#         r_min_index = int(max(r_center+offset-margin, 0))
#         r_max_index = int(min(r_center+offset+margin, image.shape[1]))
#         r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

#         # Append this layer to window_centroids
#         window_centroids.append((l_center, r_center))

#     return window_centroids
