import cv2
import numpy as np

def get_contours(threshold_img, actual_img, debug_mode=False):
    """
    This function takes the threshold image and color image.
    . and draws the contours on actual_img found from the threshold_img
    . Finds the biggest corners of the rectangle(4 cornered one) based on the maximum area covered by the rectangle(i.e., Document)
    :param threshold_img: A preprocessed threshold image (Gray scale) to find contours, corners, area..
    :param actual_img: Actual image from the camera feed to draw the contours found from the threshold_img.
    :param debug_mode: Displays additional log messages and some other extra error-detecting features if set to True, else not if set to False
    :return: The biggest corner found in the image sent.
    NOTEs
        A small note before returning the biggest_corners//
    At
    #finally return the biggest_corners detected in the image..
    # But sometimes returning the biggest_corners as [] (Just an empty list). Even validation is done before that only of length 4 should be assigned, but this is happening...
    # Got the reason, why its happening. Once please look the above condition.. we'll enter into loop only if the area>5000, if this condition
    #  fails, our empty list which we assigned before the condition is returned which is of length-0.
    # OK, problem detected, How to solve this????????????
    # Can validate the length of the biggest_corners before returning...as if ==4, return corners else simply return That sounds good..But..
    #   calling function expects something from this called function. if len==4, good, BUT, BUT., BUT if in case its 0(We return nothing right..!!)
    #   then the called function receives nothing(i.e., it overrides the previous value), the called function gets "None" value.
    # This solution again leads to a new problem, previously problem with [], now we'll get as "None" type error.
    # Another Solution..........
    # It is not the good idea, but it works for now, later find more appropriate one..
    #   Its, validate at the called function.. as
    #          If the len(biggest_corners) == 4, then only go for warping, else not..
    """
    # Required variables..
    maxArea = 0  # later modified in the loop
    biggest_corners = np.array([])
    contours, hierarchy = cv2.findContours(image=threshold_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    # Now loop through the contours..
    for contour in contours:
        # Find the area of the contour, so that it helps us in filtering unwanted noise..
        contour_area = cv2.contourArea(contour=contour)
        if contour_area > 5000:   # Playable Parameter..
            # Then draw the contours(lines) for the image, which satisfies the condition..of area...
            if debug_mode: cv2.drawContours(image=actual_img, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=3)  # contourIdx=-1 means to select all the contours..

            # Find the perimeter...
            perimeter = cv2.arcLength(curve=contour, closed=True)

            # approximating/finding the corners of the object found..
            approx_corners = cv2.approxPolyDP(curve=contour, epsilon=0.02*perimeter, closed=True)

            # As a paper document is like a rectangle with 4 corners, we'll validate based on this condition as well as
            #   there is a chance of having some images(or something like rectangular shape).
            #   So, for this we check the maximum area one(Most likely it would be the paper/document).
            if len(approx_corners) == 4 and contour_area > maxArea:
                # update the maxArea, as now area is greater than previous taken maxArea..
                maxArea = contour_area
                # get the approximated corners of the maximum area one..(as corners will also be the larger ones if the area is greater right..!!)
                biggest_corners = approx_corners

                # if debug_mode: print("Got {} approximated corners..and corners: ".format(len(approx_corners)), biggest_corners)

    # Draw the biggest corners.. uncomment the below line if would like to display the direct result..
    # cv2.drawContours(image=actual_img, contours=biggest_corners, contourIdx=-1, color=(255, 0, 0), thickness=30)  # contourIdx=-1 means select all contours..(Here select all the corner points..)
    # finally return the biggest_corners detected in the image..

    return biggest_corners


def display_concatenated_results(images, image_names):
        """
        :param images: list of images to be concatenated.
        :return: Concatenated images, in the order how the list is passed as  parameter.
        NOTEs: (1)
                    >>> lst = [[1, 2, 3],[4, 5, 6]]
                    >>> len(lst)   # Returns no. of rows..
                    2
                    >>> len(lst[0])  # returns no. of cols.. P2N--<lst[any_row_number]
                    3
                (2)
                    Concatenation can be done as
                        -> First concatenating all column images, via cv2.vconcat()
                        -> Then concatenating all the column-concatenated_images horizontally, via cv2.hconcat()
                       or in reverse manner, first concatenating the individual rows, via cv2.hconcat(), then concatenating
                          vertically via cv2.vconcat()
                    Below is the implementation for the first approach described above..
        WARNING:
                 Doesn't work, when grey scaled images need to be concatenated..
                 Solution:
                    >>> img = cv2.imread("Resources/Doraemon.jpg")  # Reading as a color image, can specify flags to read directly as converted image
                    ..
                    ..
                    ..
                    >>> grayImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)  # Converting to gray scale. Becomes 3 channeled image to 2 channeled image.
                    '''
                         A 2 channeled image cannot be concatenated with 3 channeled image.
                         Solution for this: (With help of https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/)
                    '''
                    >>> grayImg_3channeled = cv2.cvtColor(src=grayImg, code=cv2.COLOR_GRAY2BGR)  # This makes 2 channeled image to 3 channeled image
                    '''
                        Now by this method we can even concatenate the gray-scaled images with colored images.
                    '''
                    Shortcut:
                    >>> grayImg_3channeled = cv2.cvtColor(src=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY), code=cv2.COLOR_GRAY2BGR)  # First converted to 2 channeled(grayscale) from 3 chanel(color), then converted to 3 channeled(grayscale)
        Phases of execution:
                Phase-1: Validation, whether all the rows has same no. of columns or not
                Phase-2: If phase-1, succeeds.. proceeds for concatenation..
        Upgrade suggestions:
             Resizing the images with a scale factor like in https://pythonexamples.org/python-opencv-cv2-resize-image/
        """
        """PHASE-1: Validation"""
        cols = len(images[0])  # Get the no. of cols in a row, Say first row..can be any row, as all rows must have same no. of cols
        for row_count in range(len(images)):
            if len(images[row_count]) is not cols:
                # print("Received {0}Rows and {1}columns".format(images.shape[0], images.shape[1]))
                print("ERROR: All the rows must have same no. of columns, Please check the list passed..!!")

                print("Hellow")
                return None
        """PHASE-2: Concatenating Images"""
        temp_vertical_concatenated_images = []
        for col_idx in range(len(images[0])):
            temp_row_images = []  # Empty each time after concatenating..!! MUST, else "cv2.error: OpenCV(4.2.0) C:\projects\opencv-python\opencv\modules\core\src\matrix_operations.cpp:68: error: (-215:Assertion failed) src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'cv::hconcat'" error
            for row_idx in range(len(images)):
                # If the image is of gray scale(2 channels).. convert to BGR scale (2 channels) as for a gray_image len(shape) gives 2 where as for a color image it results 3
                if len(images[row_idx][col_idx].shape) == 2:
                    # print("grayscale image: at", (row_idx, col_idx))
                    # print("Converting to BGR format")
                    images[row_idx][col_idx] = cv2.cvtColor(src=images[row_idx][col_idx], code=cv2.COLOR_GRAY2BGR)
                # Add the labels..
                cv2.rectangle(img=images[row_idx][col_idx], pt1=(0, 0), pt2=(int(len(image_names[row_idx][col_idx])*100), 130), color=(255, 255, 255), thickness=cv2.FILLED)
                cv2.putText(img=images[row_idx][col_idx], text=image_names[row_idx][col_idx], org=(5, 105), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=5, color=(255, 0, 255), thickness=5)
                temp_row_images.append(images[row_idx][col_idx])
            temp_vertical_concatenated_images.append(cv2.vconcat(temp_row_images))
        return cv2.hconcat(temp_vertical_concatenated_images)

