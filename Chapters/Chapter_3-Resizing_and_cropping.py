import cv2

img = cv2.imread("Resources/lena.jpg")
print(img.shape)

# Resizing the image..
imgResize = cv2.resize(src=img, dsize=(200, 300))  # dsize=(width, height)

# Cropping the image.. for this no need of an OpenCV function, can do normally via matrix manipulation, but its bit tricky --> first height, next width not like the convention for a opencv function--> width, height
croppedImg = img[250:380, 230:370]   # [height, width]-- height from which to which (ex: h1:h2),  width from where to where (Ex: w1:w2)--- combinely.. [h1:h2, w1:w2]
# -for this stmt, here........^^^, i.e., if w2<h2, getting a black cut edge, know why this happening..??
# Displaying images...
cv2.imshow("Original Image", img)
cv2.imshow("Resized image", imgResize)
cv2.imshow("Cropped Image", croppedImg)

cv2.waitKey(0)
