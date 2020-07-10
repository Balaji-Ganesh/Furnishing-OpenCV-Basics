import cv2


number_plate_classifier = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
plate_count = 1
while True:
    # load the image..
    image = cv2.imread("Resources/number_plate_{}.jpg".format(plate_count))
    detectedImg = image.copy()
    imageROI = image.copy()
    # Convert to gray scale..
    grayImg = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Detect the number plates...
    number_plates = number_plate_classifier.detectMultiScale(grayImg, scaleFactor=1.1)

    for (x, y, width, height) in number_plates:
        # validate based on area..
        area = width*height
        # print(area, " for ", plate_count)
        if 1000 < area < 25000:
            # Draw the rectangle to show the number plate...
            cv2.rectangle(img=detectedImg, pt1=(x, y), pt2=(x+width, y+height), color=(255, 0, 255), thickness=2)
            cv2.rectangle(img=detectedImg, pt1=(x-1, y-25), pt2=(x+130, y), color=(255, 0, 255), thickness=cv2.FILLED)
            cv2.putText(img=detectedImg, text="Number Plate", org=(x, y-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
            # Extract the image..
            imageROI = image[y:y+height, x: x+width]

    cv2.imshow("Detected number plate n the actual image", detectedImg)
    cv2.imshow("Extracted number plate", imageROI)
    # press 'n' to select next image..and save the previous detected results..
    if cv2.waitKey(0) == ord('n'):
        if plate_count <= 3:

            print("Saved results.. for number plate:", plate_count)
            # For splash display on the displaying window,, if uncommented this, please also uncomment the cv2.waitKey(2000),,=---Know why this is not working..
            # cv2.rectangle(detectedImg, pt1=(0, 200), pt2=(640, 300), color=(0, 355, 0), thickness=cv2.FILLED)
            # cv2.putText(img=detectedImg, text="Result saved..!!", org=(150, 265), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5)

            cv2.imwrite("Results/extracted_number_plate_{}.jpg".format(plate_count), imageROI)
            cv2.imwrite("Results/Detected_number_plate_{}.jpg".format(plate_count), detectedImg)
            # cv2.waitKey(2000)
            plate_count += 1
        else:
            break
