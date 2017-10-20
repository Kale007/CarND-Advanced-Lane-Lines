**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./ouput_images/undistorted_chessboard.png "Undistorted Chessboard Image"
[image2]: ./ouput_images/undistorted_test_image.png "Undistorted Road Image"
[image3]: ./ouput_images/combined_threshold_test_image.png "Binary Threshold Road Image"
[image4]: ./ouput_images/transformed_test_image.png "Transformed Road Image"
[image5]: ./ouput_images/transformed_threshold_test_image.png "Transformed Binary Road Image"
[image6]: ./ouput_images/reg_of_interest_test_image.png "Transformed Region of Interest Image"
[image7]: ./ouput_images/projected_test_image.png "Projected Image"
[video1]: ./ouput_images/project_video_results.avi "Project Video Results"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step in contained in the file `calib_utils.py`. Here, I acquire the chessboard images from `main.py`. I start by setting the number of points in the x and y direction for the chessboard. I then use a helper function get a list of `objpoints` (replicated array of values) and `imgpoints` (updated upon successful calculations of corners from `cv2.findChessboardCorners`).

I then use the `cv2.calibrateCamera` function to compute camera calibration matrix `mtx` and distortion coefficients `dist`. I check to see if the chessboard was properly undistorted for every chessboard image. Image1 is an example.
![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After obtaining the camera calibration and distortion coefficients, I load the test images and pipe them through `pipeline.py`. Here I check to see if the input is a video or an image. For images, I allow each image to be treated as a new frame so that prior lane detections are independent. The image is first undistorted with `cv2.undistort` using the same `mtx` and `dist` that was obtained during camera calibration (line 25). This is an example of an undistorted test image:
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I then take the `undistorted_image` and pipe it through `threshold_utils.py` to the `perform_color_and_gradient_thresholding` function. Here, the undistorted images are binary thresholded by color and gradients. For color, I used red and saturation channel thresholding on line 41 (params: `red_thresh = (200, 255)` `saturation_thresh = (170, 255)`). For gradients, I used `gradx` and `grady` or `mag_binary` and `dir_binary` thresholding on line 56 (params: `sobel_kernel = 3`, `sobel_thresh_x = (20, 100)`, `sobel_thresh_y = (20, 100)`, `mag_thresh = (30, 100)`, `dir_thresh = (0.7, 1.3)`). This produced the following binary image:
![alt text][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The binary `combined_threshold_image` is then piped through `transform_utils.py` on line 56. Through visual analysis of the image, the following source points and desired transformation points were chosen:
`src = np.float32([[585, 450], [695, 450], [1125, 720], [200, 720]])`
`dst = np.float32([[300, 0], [1000, 0], [1000, 720], [300, 720]])`

The images go through the `image_warp` function where a perspective transform is performed with `cv2.getPerspectiveTransform` on line 15. The inverse transform matrix `Minv` is also calculated to unwarp the image later for the final result. The warped image is then returned using `cv2.warpPerspective`, and visually checked to verify parallel lanes.

Here is an example of the non-binary transformed image:
![alt text][image4]

And here is an example of a binary transformed image:
![alt text][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Back in `pipeline.py` line 65, the `transformed_threshold_image` is then piped to file `detect_lanes_utils.py`. A sliding window search using the `sliding_window_search` helper function is performed. The function will identify histogram peaks (line 19) of the `transformed_threshold_image` using the helper function `find_image_histogram_peaks`. Windows are then centered around points that the lane is detected. The `generate_polynomial` helper function is used on line 65 to fit a second-order polynomial to these windows. A region of interest for each lane is then calculated using the `visualize_region_of_interest` helper function on line 96, as shown below.
![alt text][image6]

For video inputs, the windows will be calculated using previous frames after the first frame.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was then found in file `detect_lanes_utils.py` using the `calculate_radius_of_curvature` helper function on line 99. The radius of curvature was calculated as the average of the left radius `left_curverad` and right curve radius `right_curverad` on line 75 in `pipeline.py`. The distance from center was calculated as `position_of_car - center_of_lane` in `detect_lanes_utils` line 197. The values were accurate to the real-world situation, between 300-900 m for radius of curvature, and around 0.3m away from the center of lane.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

A clearly identified lane is projected back onto the road in file `detect_lanes_utils.py` line 77 using helper function `project_onto_road`. This function will also unwarp the image using `Minv` calculated during camera calibration. The following image is the result of this, with the radius and distance from center superimposed on line 82-83 in `pipeline.py`.
![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./ouput_images/project_video_results.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The pipeline has the potential to fail in shadowy lanes or during nighttime driving. These situations offer much more planning for how to threshold the image. The pipeline attempts to minimize drastic changes in the video frames by utilizing previous inputs to calculate the lane boundaries. However, this can be disrupted for consistently bad video frames. Using more data from previous images could help with problematic road conditions by using moving averages of values to detect incremental changes. An alert system can then be established to notify the driver when there is too much deviation from normal inputs.
