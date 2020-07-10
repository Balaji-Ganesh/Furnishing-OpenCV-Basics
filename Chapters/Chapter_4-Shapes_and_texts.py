import cv2
import numpy as np

# lets create a black image (like.. to resemble a blackboard). Say with a size of 500 X 900
blackboard = np.zeros(shape=(500, 900, 3), dtype=np.uint8)

"""lets start playing.."""  # NOTE: For the opencv functions, first comes width, next height...
'''LINES'''
# lets draw a diagonals to the board..
cv2.line(img=blackboard, pt1=(0, 0), pt2=(blackboard.shape[1], blackboard.shape[0]), color=(255, 255, 255), thickness=2, lineType=3)
cv2.line(img=blackboard, pt1=(blackboard.shape[1], 0), pt2=(0, blackboard.shape[0]), color=(255, 255, 200), thickness=4)

'''RECTANGLES'''
# Lets draw a rectangles, whose diagonals will coincide with the diagonals of the blackboard..
offset_from_midpt = 100
# pt1 = ((blackboard.shape[1]//2)-offset_from_midpt, (blackboard.shape[0]//2)-offset_from_midpt)   # Will resemble the North-West (or upper-left) from the center by "offset_from_midpt" units
# pt2 = ((blackboard.shape[1]//2)+offset_from_midpt, (blackboard.shape[0]//2)+offset_from_midpt)   # Resembles the bottom left direction from center(or midpt) by "offset_from_midpt" units..
# --------------------But pt1 and pt2 making a bound to circles when drawn rectangles, !!!!DON'T DELETE THIS..!!!
pt1 = ((blackboard.shape[1]//2)-410, (blackboard.shape[0]//2) - 230)   # Will resemble the North-West (or upper-left) from the center by "offset_from_midpt" units
pt2 = ((blackboard.shape[1]//2)+410, (blackboard.shape[0]//2)+230)   # Resembles the bottom left direction from center(or midpt) by "offset_from_midpt" units..
cv2.rectangle(img=blackboard, pt1=pt1, pt2=pt2, color=(100, 230, 25), thickness=cv2.FILLED)


'''CIRCLES'''
# lets draw the circles at the centre of the diagonals(or meeting point or intersecting point or mid point of both..) of rectangle..
center = (blackboard.shape[1]//2, blackboard.shape[0]//2)  # Calculated by keeping the OpenCV's image convention in mind..
# print(center)
cv2.circle(img=blackboard, center=center, radius=200, color=(0, 255, 255), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=180, color=(255, 25, 55), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=160, color=(255, 35, 255), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=140, color=(55, 2, 255), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=120, color=(25, 5, 25), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=100, color=(255, 255, 25), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=80, color=(255, 5, 25), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=60, color=(25, 255, 25), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=40, color=(25, 95, 255), thickness=cv2.FILLED)
cv2.circle(img=blackboard, center=center, radius=3, color=(255, 0, 0), thickness=cv2.FILLED)
# ---SPECIAL-----cv2.circle(img=blackboard, center=center, radius=410, color=(0, 0, 0), thickness=cv2.FILLED)  #!!!!!!!!!!!!Making this uncomment, will make a nice frame.. Just adjust the background color..!!

'''TEXTS'''
cv2.putText(img=blackboard, text="Nice Play with OpenCV..Quite good experience..!!", org=(30, 45), fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1, color=(255, 255, 255), thickness=2)
cv2.putText(img=blackboard, text="Center of Black board", org=(center[0]+10, center[1]+10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
            color=(255, 255, 255), thickness=3)
# Display the resultant image (i.e., Black board with drawn shapes, lines, texts..)
cv2.imshow("Children's Black board", blackboard)
cv2.waitKey(0)

