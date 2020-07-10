import cv2
import numpy as np
import Projects.DocumentScanner.utils as utils

class DocumentScanner:
    def __init__(self, debug_mode=False):
        # Get the camera access..
        self.capture = cv2.VideoCapture("Resources/video_1.mp4")

        # Know the mode in which the user would like to run the program..
        self.debug_mode = debug_mode

    def get_camera_feed(self):
        read_status, frame = self.capture.read()
        return frame

    def preprocessing(self, frame):
        """
        This function will pre process the image by..
            . First converting into Gray scale
            . Next, Blurring (Gaussian Blur)
            . Apply Canny Edge detection.. (Edges might appear small) so...
            . Dilate the Edges(Edges get bigger..) for 2 iterations...
            . Erode the Edges(Edges get thinner) for 1 iteration.. this result will be with good edges..
        :return: Final image of good edges.. (Gray scaled image)
        """
        # Converting into grayscale..
        imgGray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        # Blur the image..
        gaussian_blur = cv2.GaussianBlur(src=imgGray, ksize=(5, 5), sigmaX=1)  # Playable parameters..
        # Detect the edges...
        canny_img =cv2.Canny(image=gaussian_blur, threshold1=300, threshold2=200)   # Playable parameters..
        # Dilate (Expand) the edges.. perform for 2 iterations.
        kernel = np.ones((5, 5))
        dilated_img = cv2.dilate(src=canny_img, kernel=kernel, iterations=2)
        # Now Erode (Shorter) the edges.. perform for 1 iteration
        eroded_img = cv2.erode(src=dilated_img, kernel=kernel, iterations=1)
        # cv2.imshow("eroded_img", eroded_img)
        # cv2.imshow("canny img", canny_img)
        return eroded_img

    def scan_documents(self, image):
        # Preprocess the frame..
        document = self.preprocessing(frame=image)
        return document

    def reorder_corners(self, unordered_corners):
        """
        This function will reorder the corner poits and sends to the calling function(i.e, to warp_perspective())
        Why need Reordering?
            The corners detected by the contours may not be in order which can applied for warping.
            So, those points need to ordered before we apply warp perspective as top-left, top-right, bottom-left, bottom-right.
              like this order - "pts2 = np.float32([   [0, 0],      [width, 0],       [0, height],      [width, height]])"
                                where             -^^Top-Lft-^^,-^^Top-right-^^, --^^bottom-left--^^, -^^bottom-right--^^
              which we send to the getPerspectiveTransform() to get transformation_matrix.
            If not performed re-ordering, the result may not be as desired, more curious...??? try without applying re-ordering (May work for some cases, but doesn't work if page is rotated...etc.)

        How its done?
            In the list of 4 corner pair of x and y points..to obtain the four corner points..
                (Can be understood better if tested the point-coordinates by an image in MS-Paint or any other application.)

                Top-Left :  We perform addition between the 4 pairs.
                            The one with the least value will be the Top-left pair
                Bottom-Right: We perform the addition on 4 pairs, the one with the highest value will be Bottom-right paor

                These are bit tricky, but so cool..(Look below NOTEs)
                Top-Right: Perform the difference on 4 pairs, the one with the least pair is the top-right pair.

                Bottom-left: Perform the difference on 4 pairs, the one with the highest value pair will be the bottom-left pair.


        :return: The ordered border_pairs as order of [top-left, top-right, bottom-left, bottom-right] like as we give points to cv2.getPerspectiveTransform()

        NOTEs:
            Assume a rectangle as an image and it is opened in MS-Paint, and the noted co-ordinates are as respectively:
                top-Left    :   [10, 30]
                top-right   :   [400, 30]
                bottom-left :   [10,500]
                bottom-right:   [400, 500]
                All these pairs are taken as list and performed some experiments as.....
            >>> list = [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>> list
            [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>> import numpy as np
            >>>arr = np.array(list)
            >>> arr
            array([[ 10,  30],
                   [400,  30],
                   [ 10, 500],
                   [400, 500]])
            >>>arr.argmin()  # If not mentioned axis, by defaults assumes like flattened array..
            0
            >>>arr.argmax()
            5
            >>> np.argmin(list)  # returns the index of the least pair..
            0
            >>> np.argmax(list)  #returns the index of the highest pair
            5
            # You may think now, why we need the axis. Look above, it gave "5" as index(as it assumes as flattened array) it gives IndexOutBounds error.
                So better to use axis that to row wise(denoted with 1) as we need the pair sum not the sum of all X coordinates and Y coordinates i.e., we get this if applied the operation column wise(i.e., axis=0)
            >>> list
            [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>> np.argmin(list, axis=0)
            array([0, 0], dtype=int32)  # means in the four columns, the min elements are in first row.. think again and experiment if not clearn..
            >>>np.argmin(list, axis=1)
            array([0, 1, 0, 0], dtype=int32)
            >>> np.argmax(list)
            5
            >>> list[np.argmin(list)]  # Should give the least value pair --i.e.,--> Top-left pair   # this works fine, even if nto passed axis
            [10, 30]
            >>> list[np.argmax(list)]  # Should give the highest value pair --i.e.,--> bottom-Right pair  # will get error if not passed axis
            Traceback (most recent call last):
            File "<input>", line 1, in <module>
            IndexError: list index out of range
            # Got error as not applied the operation row wise..
            >>> list
            [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>>arr.sum()  # !!!!!!!! We don't need this..
            1880
            >>> arr.sum(axis=1)  # this we need...
            array([ 40, 430, 510, 900])
            >>> add = arr.sum(axis=1)
            >>> list
            [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>>list[np.argmin(add)]  # should result the top-left pair i.e., least add-pair
            [10, 30]                                 # Yes its as expected..
            >>>list[np.argmax(add)]   # Should result the bottom-right pair i.e., highest add-pair
            [400, 500]
            >>> diff = arr.diff(axis=1)  # Don't perform like this..
            Traceback (most recent call last):
              File "<input>", line 1, in <module>
            AttributeError: 'numpy.ndarray' object has no attribute 'diff'
            >>>diff = np.diff(list, axis=1)  # Perform like this..
            >>>diff
            array([[  20],
           [-370],
           [ 490],
           [ 100]])
           # --------------Understand how difference is performed.. Main point to understand the rest part.
           #   if like [x, y] performed "y-x". This operation is applied across all the rows. Now look at the result above...Is it clear.!!
            >>>list
            [[10, 30], [400, 30], [10, 500], [400, 500]]
            >>>list(np.argmin(diff))  # sorry..!!
            Traceback (most recent call last):
            File "<input>", line 1, in <module>
            TypeError: 'list' object is not callable
            >>>list[np.argmin(diff)]  # Should result the top-right pair..
            [400, 30]                 # Yes..!! Its as expected..
            >>>list[np.argmax(diff)]  # should result the bottom-left pair
            [10, 500]                  # Yes .. !! Its as expected..
            With help of :
                1. https://numpy.org/doc/1.18/reference/generated/numpy.argmin.html
                2. https://numpy.org/doc/1.18/reference/generated/numpy.argmax.html
                3. GFG
            WARNING:
                All the corners arrays should be of type float as cv2.getPerspectiveTransform() won't work.
                Else error like "error: (-215:Assertion failed) src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4 in function 'cv::getPerspectiveTransform'" is displayed...
                We got stucked by this...and later..by help of https://stackoverflow.com/questions/9808601/is-getperspectivetransform-broken-in-opencv-python2-wrapper, could able to resolve the issue..
        """
        # print("Received unordered corners are(Not reshaped): ", unordered_corners)
        # reshape the unordered_corners (It will be of shape.-> (4, 1, 2)) reshape it to (4, 2) so that we con perform summation and difference
        unordered_corners = np.float32(unordered_corners).reshape(4, 2)
        # print("Unordered corners (Reshaped)= ", unordered_corners)
        # create a ordered_corners array with dummy values with actual shape...
        ordered_corners = np.ones(shape=(4, 2), dtype=np.float32)
        add = np.sum(unordered_corners, axis=1)    # perform row-wise operation..
        #print(unordered_corners)
        # print("add = ", add)
        diff = np.diff(unordered_corners, axis=1)
        # print("diff = ", diff)
        # print("top-left pair: ", unordered_corners[np.argmin(add)])

        # Top-left corner..
        ordered_corners[0] = unordered_corners[np.argmin(add)]
        # Bottom-right-corner..
        ordered_corners[3] = unordered_corners[np.argmax(add)]

        # Top-left corner
        ordered_corners[1] = unordered_corners[np.argmin(diff)]
        # Bottom-left corner..
        ordered_corners[2] = unordered_corners[np.argmax(diff)]

        # Return the ordered corners..
        return ordered_corners

    def warp_perspective(self, image, biggest_corners):
        # As sometimes there is a chance of getting unordered points, lets handle that..
        pts1 = self.reorder_corners(biggest_corners)
        height, width = image.shape[0], image.shape[1]
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # find the transformation matrix..
        transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # print(transformation_matrix)
        # Perform perspective...
        perspective_img = cv2.warpPerspective(src=image, M=transformation_matrix, dsize=(width, height))

        # Crop the image.. so that we can remove the excess borders..
        croppedImg = perspective_img[20:perspective_img.shape[0]-20, 20: perspective_img.shape[1]-20]
        croppedImg = cv2.resize(croppedImg, (width, height))

        # Return the result
        return croppedImg



def start_document_scan(debug_mode=False):
    # Create the object for the class "DocumentScanner"
    docScanner = DocumentScanner()

    # Scan till user quits..
    #while True:
    # Get the camera feed..
    # frame = docScanner.get_camera_feed()
    frame = cv2.imread("Resources/paper.jpg")
    #frame = cv2.resize(src=frame, dsize=(frame.shape[0]//2, frame.shape[1]//2))
    image = frame.copy()
    # Scan the document..
    document = docScanner.scan_documents(image=image)

    # find the contours.. in the image(document), and get the biggest corners in the image that resemble the document which we would like to scan..
    biggest_corners = utils.get_contours(threshold_img=document, actual_img=frame, debug_mode=debug_mode)
    # print(biggest_corners)
    # Perform warp perspective..based on the corners we get..
    # But sometimes there is a chance of getting the biggest_corners as empty list in biggest_corners, so validating.. refer the utils.get_contours() function.. for more clarity..
    warp_perspective_img = image
    if len(biggest_corners) != 0:
        warp_perspective_img = docScanner.warp_perspective(image=image, biggest_corners=biggest_corners)

    # Display results..]
    cv2.namedWindow("contourImg", cv2.WINDOW_NORMAL)
    cv2.namedWindow("document", cv2.WINDOW_NORMAL)
    cv2.imshow("contourImg", frame)
    cv2.imshow("document", document)
    cv2.namedWindow("warp_image", cv2.WINDOW_NORMAL)
    cv2.imshow("warp_image", warp_perspective_img)

    cv2.waitKey(0)
    # break

def detailed_document_scan(debug_mode=False):
    docScanner = DocumentScanner(debug_mode=debug_mode)
    original_image = cv2.imread("Resources/paper.jpg")  # got good results with paper.jpg and paper_low.jpg  (TIP: Use Windows Photos Viewer to rotate the images..)
    original_image = cv2.resize(original_image, (3000, 2900))
    image = original_image.copy()
    # Grayscaled..
    grayscaledImg = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    # Thresholded Image..
    threshold_img = docScanner.preprocessing(frame=image)

    # contours..
    contourImg = original_image.copy()
    biggest_corners = utils.get_contours(threshold_img=threshold_img, actual_img=contourImg, debug_mode=True)
    # Biggest Corners..
    corners_img = original_image.copy()
    cv2.drawContours(image=corners_img, contours=biggest_corners, contourIdx=-1, thickness=10, color=(255, 0, 0))

    # Warp perspective..
    warp_perspective_img = original_image.copy()
    if len(biggest_corners) != 0:
        warp_perspective_img = docScanner.warp_perspective(image=warp_perspective_img, biggest_corners=biggest_corners)

    # Gray scaled Warp image..
    warp_gray = cv2.cvtColor(src=warp_perspective_img, code=cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold Warp image..
    warp_adaptive_thresh = cv2.adaptiveThreshold(src=warp_gray, maxValue=200, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 thresholdType=cv2.THRESH_BINARY, C=56, blockSize=255)  # Playable parameters.. NOTE: blockSize's value must be only of type odd else error as "error: (-215:Assertion failed) blockSize % 2 == 1 && blockSize > 1 in function 'cv::adaptiveThreshold'"  with help at https://stackoverflow.com/questions/27268636/assertion-failed-blocksize-2-1-blocksize-1-in-cvadaptivethreshold could able to resolve...
    # Warp canny..
    warp_canny = cv2.Canny(warp_gray, threshold1=100, threshold2=200)

    # Warp dilated..
    kernel = np.ones((5, 5))
    warp_dilated = cv2.dilate(warp_canny, kernel=kernel, iterations=2)
    warp_eroded = cv2.erode(warp_dilated, kernel=kernel, iterations=2)

    concat_images = [[original_image, grayscaledImg, threshold_img, contourImg, corners_img],
                     [warp_perspective_img.copy(), warp_gray, warp_adaptive_thresh, warp_canny, warp_eroded]]
    concat_images_names = [["original_image", "Gray scaled", "Threshold", "Contours", "Biggest Contours"],
                           ["Warp perspective", "Warp Gray", "Warp Adaptive Threshold", "Warp canny", "Warp eroded"]]

    concat_img = utils.display_concatenated_results(concat_images, concat_images_names)

    # Display results..
    cv2.namedWindow("Work flow ...", cv2.WINDOW_NORMAL)
    cv2.imshow("Work flow ...", concat_img)
    cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Image", warp_perspective_img)
    if cv2.waitKey(0) == ord('s'):  # 's' for save..
        cv2.imwrite("Results/Final image.jpg", warp_perspective_img)
        cv2.imwrite("Results/Work flow.jpg", concat_img)
        print("Results saved successfully")

if __name__ == '__main__':
    #start_document_scan(debug_mode=True)
    detailed_document_scan(debug_mode=True)


"""
LOG: 
    Ran on 8th july, 2020, worked successfully with dimensions (200, 300) and no Memory over flow till (3000, 3000)
    
    Getting "error: (-4:Insufficient memory) Failed to allocate 365783040 bytes in function 'cv::OutOfMemoryError'" if not resized.. Solve
"""

"""
NOTE: All the playable and adjustable parameters must be tuned as per the image..
        The current values made good result for the Resources/paper.jpg  -- image by mentor: MurtazaHassan Workshop
    When tried for our custom images, they didn't work well due to low clarity...
    later we'll try with more better clarity images.. ------------NOTE on 8th July, 2020
"""