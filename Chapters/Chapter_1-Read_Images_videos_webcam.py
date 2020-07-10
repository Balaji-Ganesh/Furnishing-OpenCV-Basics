import cv2

"""
# Lesson-1: Reading Images
test_img = cv2.imread("Resources/lena.jpg")   # Read the image
cv2.imshow("Image", test_img)                
 
 # Display the image
cv2.waitKey(0)                                # How much time the image should be displayed, 0-- infinite time(or till a key press), a number -- denotes time in milliseconds
"""

# Lesson-2: Working with videos
# capture = cv2.VideoCapture("Resources/vtest.avi")  # Getting the video from source, or can also be a number that denotes the webcam count.!!(starting from 0)
capture = cv2.VideoCapture(0)
# Setting additional properties..refer https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=videocapture#bool%20VideoCapture::set(int%20propId,%20double%20value) for additional properties.. The sequence count of the property name can also be used instead of the property name.. (Don't forget to remove the prefix "CV_" before all the propid's, credits: https://www.thetopsites.net/article/58272735.shtml)
print("Actual Properties of the captured video: ")
frame_width = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Equivalent to pass the sequence-count of cv2.CV_CAP_PROP_FRAME_HEIGHT-4
frame_height = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # equivalent to pass the sequence-count of cv2.CV_CAP_PROP_FRAME_WIDTH-3
print("Dimensions of frames : {0} X {1}".format(frame_height, frame_width))
print("FPS                  : ", capture.get(cv2.CAP_PROP_FPS))
print("Brightness           : ", capture.get(cv2.CAP_PROP_BRIGHTNESS))
print("Contrast             : ", capture.get(cv2.CAP_PROP_CONTRAST))

# Set the frame width and height to custom values..
print("\nModified Properties: ")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # If not suitable, then adjusted by itself (i.e., downgraded..!!)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
print("Dimensions of frames : {0} X {1}".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

# Setting the brightness..
capture.set(cv2.CAP_PROP_BRIGHTNESS, 200)
print("Brightness           : ", capture.get(cv2.CAP_PROP_BRIGHTNESS))

# Setting the contrast..
capture.set(cv2.CAP_PROP_CONTRAST, 40)
print("Contrast             : ", capture.get(cv2.CAP_PROP_CONTRAST))


# Display the video frames with modified properties..
while True:                            # Keep on reading the images from video infinitely..
    read_status, img = capture.read()  # Read the sequence of frames
    cv2.imshow("Video", img)           # Display the image
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Stop displaying when pressed 'Q' Key..
        break                          # Breaks the infinite loop

