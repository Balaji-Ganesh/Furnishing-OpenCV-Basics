import cv2

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


'''
LOG: Another new version is implemented in project-Document Scanner please refer the utils.py in DocumentScanner project
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
                    """
                         A 2 channeled image cannot be concatenated with 3 channeled image.
                         Solution for this: (With help of https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/)
                    """
                    >>> grayImg_3channeled = cv2.cvtColor(src=grayImg, code=cv2.COLOR_GRAY2BGR)  # This makes 2 channeled image to 3 channeled image
                    """
                        Now by this method we can even concatenate the gray-scaled images with colored images.
                    """
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

'''