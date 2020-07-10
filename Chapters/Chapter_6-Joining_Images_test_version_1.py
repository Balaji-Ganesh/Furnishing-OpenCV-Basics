import cv2

img = cv2.imread("Resources/Doraemon.jpg")
cv2.resize(img, (0, 0), None, .25, .25)

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 3 channels to 2 channels -- Cannot be stacked or concatenated.. either vertically or horizontally
# grayImg = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR) # 2 channels to 3 channels, No can be stacked...either vertically or horizontally..
# Above two LOC can be converted to one line as..
grayImg = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # Converting into Gray scale (Becomes 3 channels to 2 channels), again converting Gray to BGR (Becomes 2 channels to 3 channels)
vertical_stack_images = cv2.vconcat([img, img, img])
horizontal_stack_images = cv2.hconcat([img, grayImg, img])

cv2.imshow("Image", img)
cv2.namedWindow("Vertical Stack", cv2.WINDOW_NORMAL)   # Setting as cv2.WINDOW_NORMAL flag, will enable us to resize the window of displaying image..
cv2.imshow("Vertical Stack", vertical_stack_images)
cv2.imshow("Horizontal stack images", horizontal_stack_images)
cv2.waitKey(0)