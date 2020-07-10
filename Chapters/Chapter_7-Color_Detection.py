import cv2
import numpy as np

""" Required functions """


# Dummy function.. just for the sake of Trackbars
def dummy(dummy):
    pass


def concatenate_images(images):
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
    cols = len(images[0])       # Get the no. of cols in a row, Say first row..can be any row, as all rows must have same no. of cols
    for row_count in range(len(images)):
        if len(images[row_count]) is not cols:
            print("ERROR: All the rows must have same no. of columns, Please check the list passed..!!")
            return None
    """PHASE-2: Concatenating Images"""
    temp_vertical_concatenated_images = []
    for col_idx in range(len(images[0])):
        temp_row_images = []  # Empty each time after concatenating..!! MUST, else "cv2.error: OpenCV(4.2.0) C:\projects\opencv-python\opencv\modules\core\src\matrix_operations.cpp:68: error: (-215:Assertion failed) src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'cv::hconcat'" error
        for row_idx in range(len(images)):
            temp_row_images.append(images[row_idx][col_idx])
        temp_vertical_concatenated_images.append(cv2.vconcat(temp_row_images))
    return cv2.hconcat(temp_vertical_concatenated_images)


""" Main code starts from here.."""
# Create the trackbars..
cv2.namedWindow(winname="Adjustment Trackbars")
cv2.resizeWindow("Adjustment Trackbars", 600, 250)
cv2.createTrackbar("HUE min", "Adjustment Trackbars", 0, 255, dummy)  # count should be 180 as per video
cv2.createTrackbar("HUE max", "Adjustment Trackbars", 255, 255, dummy)
cv2.createTrackbar("SAT min", "Adjustment Trackbars", 0, 255, dummy)
cv2.createTrackbar("SAT max", "Adjustment Trackbars", 255, 255, dummy)
cv2.createTrackbar("VAL min", "Adjustment Trackbars", 0, 255, dummy)
cv2.createTrackbar("VAL max", "Adjustment Trackbars", 255, 255, dummy)


# Read the original image
img = cv2.imread(filename="Resources/opencv-logo.png")
img = cv2.resize(img, (int(img.shape[0]//2), int(img.shape[1]//2)))

# Convert to HSV
HSVimg = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)

# As we modify the values of track bar, the image should be updated as we change, so running in loop..
while True:
    # Get the positions of track bars..
    hue_min = cv2.getTrackbarPos("HUE min", "Adjustment Trackbars")
    hue_max = cv2.getTrackbarPos("HUE max", "Adjustment Trackbars")
    sat_min = cv2.getTrackbarPos("SAT min", "Adjustment Trackbars")
    sat_max = cv2.getTrackbarPos("SAT max", "Adjustment Trackbars")
    val_min = cv2.getTrackbarPos("VAL min", "Adjustment Trackbars")
    val_max = cv2.getTrackbarPos("VAL max", "Adjustment Trackbars")

    # Create a mask with the lower and upper boundaries of HSV values..
    lower_boundary = np.array([hue_min, sat_min, val_min])
    upper_boundary = np.array([hue_max, sat_max, val_max])
    maskImg = cv2.inRange(HSVimg, lowerb=lower_boundary, upperb=upper_boundary)

    # Merge the mask_img with the original image..
    final_image = cv2.bitwise_and(src1=img, src2=img, mask=maskImg)

    # Display images
    # cv2.imshow("Original Image", img)
    # cv2.imshow("HSV Image", HSVimg)
    # cv2.imshow("Masked Image", maskImg)
    # cv2.imshow("Final Image", final_image)
    concatenated_images = concatenate_images([[img, HSVimg],
                                              [cv2.cvtColor(maskImg, cv2.COLOR_GRAY2BGR), final_image]])
    cv2.imshow("Color Detection", concatenated_images)

    if cv2.waitKey(1) == 27:
        print("Final HSV values:\nLower Boundary: ", lower_boundary, "Upper Boundary: ", upper_boundary)
        break
"""
LOG:
    For detecting the O: Lower Boundary:  [  0 253   0] Upper Boundary:  [ 48 255 255]
    For Detecting the C: Lower Boundary:  [  4 255   0] Upper Boundary:  [ 70 255 255]
    For Detecting the V: Lower Boundary:  [ 71 253 255] Upper Boundary:  [255 255 255]
    For Detecting the OpenCV: Lower Boundary:  [0 0 0] Upper Boundary:  [  0   0 251]
"""