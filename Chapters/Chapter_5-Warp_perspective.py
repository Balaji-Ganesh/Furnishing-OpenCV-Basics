import cv2
import numpy as np

"""img = cv2.imread("Resources/design_box_cropped.jpg")
print(img.shape)
width, height = 1199, 809

pts1 = np.float32([[84, 258],
                   [709, 18],
                   [400, 468],
                   [1037, 188]
                   ])
pts2 = np.float32([[0, 0],
                   [width, 0],
                   [0, height],
                   [width, height]
                   ])
# print(pts1.shape)
transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
# print(transformation_matrix)
warpedImg = cv2.warpPerspective(src=img, M=transformation_matrix, dsize=(width, height))

cv2.imshow("Original Image", img)
cv2.imshow("Warped Image", warpedImg)
cv2.waitKey(0)"""

capture = cv2.VideoCapture("Resources/video_1.mp4")

while True:
    read_status, frame = capture.read()

    height, width = frame.shape[0], frame.shape[1]

    pts1 = np.float32([[[202, 95]],

                       [[973, 19]],
                      [[177, 632]],
                       [[1001, 649]]])
    pts2 = np.float32([[0, 0],
                       [width, 0],
                       [0, height],
                       [width, height]
                       ])
    # print(pts1.shape)
    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # print(transformation_matrix)
    warpedImg = cv2.warpPerspective(src=frame, M=transformation_matrix, dsize=(width, height))

    cv2.imshow("hello", frame)
    cv2.imshow("warp", warpedImg)

    if cv2.waitKey(1) == 27:
        break
