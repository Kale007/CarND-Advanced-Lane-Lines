import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
cap = cv2.VideoCapture(0)

from transform_utils import image_warp
from threshold_utils import perform_color_and_gradient_thresholding
from detect_lanes_utils import perform_detect_lanes

def pipeline(images, mtx, dist, is_video, video_file):
    # perform image steps if input is not a video
    if is_video == False:
        # Indicate this is first frame in video for lane detection
        is_first_frame = True

        left_fit = []
        right_fit = []
        for frame in images:
            # Read in each image
            image = mpimg.imread(frame)


            ### Use calibrate_and_undistort helper function to obtain undistorted image
            undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

            # plt.imshow(undistorted_image)
            # plt.show()
            # undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('../output_images/undistorted_test_image.png', undistorted_image)


            ### Transform image to top-down
            transformed_image, Minv = image_warp(undistorted_image)

            # plt.imshow(transformed_image)
            # plt.show()
            # # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('../output_images/transformed_test_image.png', transformed_image)

            # gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
            # plt.imshow(gray)
            # plt.show()


            ### Color and gradient threshold image
            combined_threshold_image = perform_color_and_gradient_thresholding(undistorted_image)

            # plt.imshow(combined_threshold_image, cmap='gray')
            # plt.show()
            # combined_threshold_image = combined_threshold_image * 255
            # cv2.imwrite('../output_images/combined_threshold_test_image.png', combined_threshold_image)


            ### Transform threshold image to top-down
            transformed_threshold_image, MinV = image_warp(combined_threshold_image)

            # plt.imshow(transformed_threshold_image, cmap='gray')
            # plt.show()
            # transformed_threshold_image = transformed_threshold_image * 255
            # cv2.imwrite('../output_images/transformed_threshold_test_image.png', transformed_threshold_image)


            ### Use sliding window search to detect lane lines
            reg_of_interest_image, projected_image, left_fit, right_fit, left_curverad, right_curverad, distance_from_center = perform_detect_lanes(transformed_threshold_image, is_first_frame, left_fit, right_fit, Minv, undistorted_image)
            
            # plt.imshow(reg_of_interest_image)
            # plt.show()
            # cv2.imwrite('../output_images/reg_of_interest_test_image.png', reg_of_interest_image)

            # plt.imshow(projected_image)
            # plt.show()
            # projected_image = cv2.cvtColor(projected_image, cv2.COLOR_BGR2RGB)

            average_radius_of_curvature =  (left_curverad + right_curverad) / 2
            projected_curvature_text = "Radius of Curvature: " + str(average_radius_of_curvature) + "m."
            projected_dist_from_center_text = "Distance from center: " + str(distance_from_center) + "m."

            print(projected_curvature_text)
            print(projected_dist_from_center_text)

            # cv2.putText(projected_image, projected_curvature_text, (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # cv2.putText(projected_image, projected_dist_from_center_text, (10,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # cv2.imshow('projected image', projected_image)
            # cv2.imwrite('../output_images/projected_test_image.png', projected_image)


    if is_video:
        is_first_frame = True

        left_fit = []
        right_fit = []

        cap = cv2.VideoCapture(video_file)

        # To save video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_file = cv2.VideoWriter('../output_images/project_video_results.avi',fourcc, 20.0, (1280, 720), isColor=True)


        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image = frame
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                transformed_image, Minv = image_warp(undistorted_image)
                combined_threshold_image = perform_color_and_gradient_thresholding(undistorted_image)
                transformed_threshold_image, MinV = image_warp(combined_threshold_image)
                reg_of_interest_image, projected_image, left_fit, right_fit, left_curverad, right_curverad, distance_from_center = perform_detect_lanes(transformed_threshold_image, is_first_frame, left_fit, right_fit, Minv, undistorted_image)

                average_radius_of_curvature =  (left_curverad + right_curverad) / 2
                projected_curvature_text = "Radius of Curvature: " + str(average_radius_of_curvature) + "m."
                projected_dist_from_center_text = "Distance from center: " + str(distance_from_center) + "m."
                print(projected_curvature_text)
                print(projected_dist_from_center_text)

                is_first_frame = False

                # Display the resulting frame with text
                cv2.putText(projected_image, projected_curvature_text, (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(projected_image, projected_dist_from_center_text, (10,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Write Video
                # save_file.write(projected_image)

                # cv2.imshow('frame', projected_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # When everything done, release the capture
        cap.release()
        save_file.release()
        cv2.destroyAllWindows()