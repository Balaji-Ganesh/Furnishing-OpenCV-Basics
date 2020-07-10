import cv2
import numpy as np

"""Required functions"""
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


def draw_contours(gaussian_canny_img, actual_img):
    # Get the contours from the gaussian blurred-canny-edged image...
    contours, hierarchy = cv2.findContours(image=gaussian_canny_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)  # Playable Parameters..
    # Get the area, calculate the edges for a shape, draw the boundary box around the shape..
    for contour in contours:
        # Find the area of the contours detected...
        area = cv2.contourArea(contour=contour)
        #print(area)

        if area > 500:  # Not required, but it eliminates the unnecessary noise..
            # Draw the detected contours..
            contour_img = cv2.drawContours(image=actual_img, contours=contours, contourIdx=-1, color=(255, 255, 0), thickness=2)  # , hierarchy=hierarchy)  # contourIdx = -1 means to select all the contours

            # Finding the perimeter of the objects detected..
            perimeter = cv2.arcLength(curve=contour, closed=True)
            #print(perimeter)

            # Approximating/Finding the corners of the image from the obtained contours
            corners = cv2.approxPolyDP(curve=contour, epsilon=0.02*perimeter, closed=True)  # Playable parameter: epsilon
            #print(corners)
            print(len(corners))
            objectCorners = len(corners)

            # Drawing a bounded rectangle around the detected object...
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(img=actual_img, pt1=(x, y), pt2=(x+width, y+height), color=(255, 25, 255), thickness=4)

            if objectCorners == 3:
                objectType = "Triangle"
            elif objectCorners == 4:  # then it can be a square (length==width or length/width=1), or a rectangle(length!=width or length/width>1)
                ratio = height/width
                if 0.95 < ratio < 1.5: # Assumed a deviation of +-10  this stmt equivalent to "if ratio > 0.95 and ratio <1.5"
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objectCorners > 4:
                objectType = "Circle"
            else:
                objectType = "None"

            #cv2.rectangle(img=actual_img, pt1=(x, y), pt2=(x+140, y-30), color=(255, 25, 255), thickness=3)
            cv2.putText(img=actual_img, text=objectType, org=(x, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=(0, 0, 0), thickness=4)
    return contour_img



"""Main code starts from here.."""
# Read the image
img = cv2.imread("Resources/shapes.png")
# Resize the image
dsize=(int(img.shape[0]//0.6), int(img.shape[1]//0.6))
img = cv2.resize(img, dsize=dsize)
# Convert to gray scale
grayImg = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# Applying Gaussian Blur filter
gaussianImg = cv2.GaussianBlur(src=grayImg, ksize=(9, 9), sigmaX=1)  # !!! Playable parameters
# Applying the Canny Edge detection technique..
cannyImg = cv2.Canny(image=gaussianImg, threshold1=50, threshold2=50)

contour_img = draw_contours(gaussian_canny_img=cannyImg, actual_img=img.copy())
# Display all the images..

concat_images = [[img,
                  cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR),
                  cv2.cvtColor(gaussianImg, cv2.COLOR_GRAY2BGR)],
                 [cv2.cvtColor(cannyImg, cv2.COLOR_GRAY2BGR),
                  contour_img,
                  np.zeros_like(img)
                  ]]

# Displaying results
concat_images = concatenate_images(concat_images)
cv2.namedWindow("Contour images", cv2.WINDOW_NORMAL)
cv2.imshow("Contour images", concat_images)
# Saving the results
cv2.imwrite("Chapter_8-Contour_Shape_Detection_1.jpg", concat_images)
cv2.imwrite("Chapter_8-Contour_Shape_Detection_2.jpg", contour_img)
cv2.waitKey(0)
