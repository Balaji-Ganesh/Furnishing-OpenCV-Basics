import numpy as np
import cv2

kernel = np.ones((5, 5))  # Used for Dilation process

''' Reading the original Image '''
img = cv2.imread("Resources/lena.jpg")

''' Converting into gray image..'''
grayImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

''' Blurring the image..'''
blurImg_color = cv2.GaussianBlur(src=img, ksize=(7, 7), sigmaX=0)  # kernelsize (ksize) denote the size of kernel, should be only odd numbered dimensions!! we usually take as square matrix dimensions, like 7X7, 9X9... rather than 11X3, 9X11..etc
blurImg_gray = cv2.GaussianBlur(src=grayImg, ksize=(7, 7), sigmaX=0)

''' Find the edges in the image..'''
cannyImg_color = cv2.Canny(image=img, threshold1=100, threshold2=200)    # Result will be as same as cannyImg_gray..!!!
cannyImg_gray = cv2.Canny(image=grayImg, threshold1=100, threshold2=200)

''' Dilating ..'''
# i.e., Expanding the edges detected in canny image edge detection.. (For now using canny, but can use any edge detector..)
dilatedImg_color = cv2.dilate(src=img, kernel=kernel, iterations=0)     # Expanding depends on value of "iterations" parameter..!
dilatedImg_gray = cv2.dilate(src=grayImg, kernel=kernel, iterations=1)
# with edges..
dilatedImg_canny_color = cv2.dilate(src=cannyImg_color, kernel=kernel, iterations=1)  # Result will be as same as dilatedImg_canny_gray..!!
dilatedImg_canny_gray = cv2.dilate(src=cannyImg_gray, kernel=kernel, iterations=1)

''' Erosion or Erode'''
# i.e., opposite of Dilation. i.e., Might be shrinking...oOo....
# on original color and gray scaled images
erodedImg_color = cv2.erode(src=img, kernel=kernel)
erodedImg_gray = cv2.erode(src=grayImg, kernel=kernel)
# on dilated images
erodedImg_dilated_color = cv2.erode(src=dilatedImg_color, kernel=kernel)
erodedImg_dilated_gray = cv2.erode(src=dilatedImg_gray, kernel=kernel)
# on edge detected + Dilated..
erodedImg_dilated_canny_color = cv2.erode(src=dilatedImg_canny_color,kernel=kernel)
erodedImg_dilated_canny_gray = cv2.erode(src=dilatedImg_canny_gray, kernel=kernel)

''' Display the resultant Images.. '''
cv2.imshow("Original Color Image", img)
cv2.imshow("Original Gray Image", grayImg)

# Blurred Images
cv2.imshow("Gaussian Blur Image - Colored", blurImg_color)
cv2.imshow("Gaussian Blur Image - Gray scaled", blurImg_gray)

# Edges detected image..
cv2.imshow("Canny edged image - color", cannyImg_color)
cv2.imshow("Canny edged image - gray", cannyImg_gray)

# Dilated Images...
cv2.imshow("Dilated image - color", dilatedImg_color)
cv2.imshow("Dilated image - gray", dilatedImg_gray)

cv2.imshow("Dilated canny image - color", dilatedImg_canny_color)
cv2.imshow("Dilated canny image - gray", dilatedImg_canny_gray)

# Eroded image
cv2.imshow("Eroded Image - color", erodedImg_color)   # looks like sketch-pen drawing
cv2.imshow("Eroded Image - gray", erodedImg_gray)
cv2.imshow("Eroded  dilated Image - color", erodedImg_dilated_color)   # looks like sketch-pen drawing
cv2.imshow("Eroded dilated Image - gray", erodedImg_dilated_gray)
cv2.imshow("Eroded dilated canny Image - color", erodedImg_dilated_canny_color)  # -----Quite Interesting...!!
cv2.imshow("Eroded dilate canny Image - gray", erodedImg_dilated_canny_gray)

cv2.waitKey(0)
