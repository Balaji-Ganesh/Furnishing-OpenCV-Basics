import cv2

classifier = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
# Set some properties..
capture.set(cv2.CAP_PROP_BRIGHTNESS, 215)
capture.set(cv2.CAP_PROP_CONTRAST, 40)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
while True:
    # Read the feed from the Web camera
    read_status, frame = capture.read()

    # Convert to Gray scale..
    grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    # Detect the faces..
    faces = classifier.detectMultiScale(grayFrame)
    # Draw the contours around the faces detected..
    for (x, y, width, height) in faces:
        # Filter the unnessary ones i.e., noise..
        if width > 200 and height > 200:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+width, y+height), color=(255, 0, 255), thickness=3)
            cv2.rectangle(img=frame, pt1=(x, y-40), pt2=(x+80, y), color=(255, 0, 255), thickness=cv2.FILLED)
            cv2.putText(img=frame, text="Face", org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow("Detected Faces", frame)

    if cv2.waitKey(1) == 27:
        cv2.imwrite("Chapter_9-Face_Detection.jpg", frame)
        break

