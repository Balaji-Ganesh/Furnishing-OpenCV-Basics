import cv2
import numpy as np

'''Required functions'''


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
    """
    """PHASE-1: Validation"""
    cols = len(images[0])       # Get the no. of cols in a row, Say first row..can be any row, as all rows must have same no. of cols
    for row_count in range(len(images)):
        if len(images[row_count]) is not cols:
            print("ERROR: All the rows must have same no. of columns, Please check the list passed..!!")
            return None
    """PHASE-2: Concatenating Images"""
    temp_row_images = []
    temp_col_images = []
    temp_vertical_concatenated_images = []
    for col_idx in range(len(images[0])):
        temp_row_images = []  # Empty each time after concatenating..!! MUST, else "cv2.error: OpenCV(4.2.0) C:\projects\opencv-python\opencv\modules\core\src\matrix_operations.cpp:68: error: (-215:Assertion failed) src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'cv::hconcat'" error
        for row_idx in range(len(images)):
            # print("Stacking the {0} row's {1} column image..".format(row_idx, col_idx), end="-->")
            # print(images[row_idx][col_idx])
            temp_row_images.append(images[row_idx][col_idx])
        # print(temp_row_images)
        temp_vertical_concatenated_images.append(cv2.vconcat(temp_row_images))
    return cv2.hconcat(temp_vertical_concatenated_images)
    # print("Final temp_row images = ", temp_row_images)


# Main code starts from here..
img = cv2.imread("Resources/Doraemon_1.jpg")
grayImg = cv2.cvtColor(src=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY), code=cv2.COLOR_GRAY2BGR)
concatenated_img = [[img, img, grayImg, img, img],
                    [grayImg, img, img, grayImg, img]]
concatenated_img = concatenate_images(concatenated_img)
cv2.imshow("Image", img)
# print(concatenated_img)
# Adding resizing window option to the concatenated images window..
cv2.namedWindow("Concatenated images", cv2.WINDOW_FREERATIO)
cv2.imshow("Concatenated images", concatenated_img)
cv2.waitKey(0)

