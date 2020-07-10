# Import the required libraries
import cv2
import numpy as np
import Projects.VirtualCanvas.utils as utils


class VirtualCanvas:
    def __init__(self, num_markers=2, debug_mode=False):
        # Mode in which the user would like to run the program..
        self.debug_mode = debug_mode  # False for normal_run, True for debug_mode
        # Get the camera access..
        self.capture = cv2.VideoCapture(0)
        # Adjust the camera capture properties..
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, 200)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        # No. of markers the user would like to draw..
        self.num_markers = num_markers
        self.marker_path_points = []  # of format -> [[x, y], colorId]

        self.markers_HSV, self.marker_colors = utils.load_data(num_markers=num_markers, debug_mode=debug_mode)
        if debug_mode: print("Data loaded Successfully.. as: markers_HSV = \n", self.markers_HSV, "\nmarker_colors = \n", self.marker_colors)

    def get_camera_feed(self):
        """
        This function will return the frames from the web cam feed
        :return:
        """
        # get the frame..from cam feed
        read_status, self.frame = self.capture.read()
        return self.frame

    def detect_and_draw_as_marker(self, image):
        """
        This function is made for testing purposes.
        The part of this function's code is used in some other functions with some optimizations

        :return:
        """
        # Required variables
        count = 0
        # convert to HSV.. so that we can filter out the image from our captured HSV values for our markers previously..
        HSVimg = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        # loop through all marker's HSV values
        for marker_HSV in self.markers_HSV:
            lower_boundary = np.array(marker_HSV[0])
            upper_boundary = np.array(marker_HSV[1])
            # Get the mask image that satisfies the lower and upper HSV values..
            maskImg = cv2.inRange(src=HSVimg, lowerb=lower_boundary, upperb=upper_boundary)

            '''Draw the contours for the mask image detected, marker point for the marker'''
            # Get the bounding box corners (In the function call to self.draw_contours(), contours are drawn to original camera feed, if self.debug_mode is set to 1)
            x, y, width, height = self.draw_contours(image, maskImg)
            if self.debug_mode:
                cv2.rectangle(img=image, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 255), thickness=3)
            # Select the marker point..
            marker_point_center = (x + width // 2, y)
            # Draw the marker point..
            # cv2.circle(img=image, center=marker_point_center, radius=5, color=(2, 255, 10), thickness=cv2.FILLED)
            cv2.circle(img=image, center=marker_point_center, radius=5, color=list(self.marker_colors[count]), thickness=cv2.FILLED)

            # Append the trace point of marker..
            self.marker_path_points.append([marker_point_center, count])
            #print(count, end="\n")
            count += 1

    def draw_contours(self, image, maskImg):
        """
        This function will find the contours for the mask image which is obtained via HSV values filtering.
        and it also draws the contours for the markers on color image of camera feed(But can be turned of by commenting the line cv2.drawContours() or if self.debug_mode is set to 0).
        :param image: Original color image of camera feed.
        :param maskImg: Mask Image (Contains only the markers)
        :return: The bounding box  corners of the detected markers..
        """
        # Required variables..
        x, y, width, height = 0, 0, 0, 0
        # Find contours..
        contours, hierarchy = cv2.findContours(image=maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)  # Playable Parameters..
        # Draw the contours..
        for contour in contours:
            # Calculate the area of the contour, so can remove unnecessary contours..
            area = cv2.contourArea(contour=contour)
            if area > 3000:  # Playable adjustment..!! Found Good as 3000 for current light condition.. change this if light condition changes..
                # Draw the contours to the image -- actual frame..
                if self.debug_mode:
                    cv2.drawContours(image=image, contours=contour, contourIdx=-1, color=(255, 255, 0), thickness=4)
                # Find the perimeter of the markers detected...
                perimeter = cv2.arcLength(curve=contour, closed=True)
                # Approximating/Finding the corners of the image from the obtained corners..
                approx_corners = cv2.approxPolyDP(curve=contour, epsilon=0.02 * perimeter, closed=True)
                # Find the bounding box rectangle for the approximated corners..
                x, y, width, height = cv2.boundingRect(approx_corners)
                # Return the values with which a rectangle can be drawn..
        return x, y, width, height

    def get_markers_center(self, image):
        """
        The code in this function is some part of the "detect_and_draw_as_marker()"
        This function will be called by trace_marker_path()
        :return:
        """
        # Required variables..
        x, y, width, height = 0, 0, 0, 0
        # convert to HSV.. so that we can filter out the image from our captured HSV values for our markers previously..
        HSVimg = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        # loop through all marker's HSV values
        for marker_HSV in self.markers_HSV:
            lower_boundary = np.array(marker_HSV[0])
            upper_boundary = np.array(marker_HSV[1])
            # Get the mask image that satisfies the lower and upper HSV values..
            maskImg = cv2.inRange(src=HSVimg, lowerb=lower_boundary, upperb=upper_boundary)

            '''Draw the contours for the mask image detected, marker point for the marker'''
            # Get the bounding box corners (In the function call to self.draw_contours(), contours are drawn to original camera feed, if self.debug_mode is set to 1)
            x, y, width, height = self.draw_contours(image, maskImg)
            if self.debug_mode:
                cv2.rectangle(img=image, pt1=(x, y), pt2=(x + width, y + height), color=(0, 0, 0), thickness=3)
            # Select the marker point..
            # marker_point_center = [x+width//2, y]
        return x + width // 2, y

    def trace_marker_path(self, image):
        """
        This function will trace the path of the marker where marker has moved..
        :return: Nothing
        """
        for trace_point in self.marker_path_points:
            cv2.circle(img=image, center=(trace_point[0]), radius=10, color=self.marker_colors[trace_point[1]], thickness=cv2.FILLED)



def drawOnCanvas(debug_mode=False):
    # Number of markers user would like to choose..
    markers_count = 3
    while markers_count <= 1:
        markers_count = int(input("How many markers would you like to use? (>1): "))
    # Create object to class "Virtual Canvas"
    virtualCanvas = VirtualCanvas(num_markers=markers_count, debug_mode=debug_mode)
    while True:
        # Get the cam feed..
        image = virtualCanvas.get_camera_feed()
        # Get the marker drawing points and save it in the marker_path_points..as [center(x, y), marker_color(count)]
        virtualCanvas.detect_and_draw_as_marker(image)
        # Draw all the path points to resemble like drawing on canvas..
        virtualCanvas.trace_marker_path(image)

        # Display the final results..
        cv2.imshow("Virtual Canvas", image)
        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()


if __name__ == "__main__":
    # mode = int(input("Debug mode -- 1 or Normal Run ---0: "))
    drawOnCanvas(debug_mode=True)

"""
LOG: 
Final HSV values as: HUE_min, SAT_min, VAL_min, HUE_max, SAT_max, VAL_max 103 45 0 120 255 255
Final HSV values as: HUE_min, SAT_min, VAL_min, HUE_max, SAT_max, VAL_max 0 80 148 255 255 255
Final HSV values as: HUE_min, SAT_min, VAL_min, HUE_max, SAT_max, VAL_max 0 89 178 255 238 255

for Dark blue, Orange, Yellow Sparx pens respectively..
"""
"""
Improvements:
    Add the facility of saving the HSV values  -- either via numpy or pkl
"""

"""
Backup code of def detect_and_draw_as_marker(self):
 """
"""
        This function is made for testing purposes.
        The part of this function's code is used in some other functions with some optimizations

        :return:
        """
"""
        while True:
            # Required variables
            count = 0
            # Get camera feed..
            image = self.get_camera_feed()
            # convert to HSV.. so that we can filter out the image from our captured HSV values for our markers previously..
            HSVimg = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
            # loop through all marker's HSV values
            for marker_HSV in self.markers_HSV:
                lower_boundary = np.array(marker_HSV[0])
                upper_boundary = np.array(marker_HSV[1])
                # Get the mask image that satisfies the lower and upper HSV values..
                maskImg = cv2.inRange(src=HSVimg, lowerb=lower_boundary, upperb=upper_boundary)

                '''Draw the contours for the mask image detected, marker point for the marker'''
                # Get the bounding box corners (In the function call to self.draw_contours(), contours are drawn to original camera feed, if self.debug_mode is set to 1)
                x, y, width, height = self.draw_contours(image, maskImg)
                if self.debug_mode:
                    cv2.rectangle(img=image, pt1=(x, y), pt2=(x+width, y+height), color=(0, 0, 0), thickness=3)
                # Select the marker point..
                marker_point_center = (x+width//2, y)
                # Draw the marker point..
                # cv2.circle(img=image, center=marker_point_center, radius=5, color=(2, 255, 10), thickness=cv2.FILLED)
                cv2.circle(img=image, center=marker_point_center, radius=5, color=self.marker_colors[count], thickness=cv2.FILLED)
                count += 1
                cv2.imshow("Virtual Canvas", image)
                #print("Working....")
            if cv2.waitKey(1) == 27:
                break
"""

"""
0 26 255 255 255 255  # orange
101 35 0 255 255 255  # Blue
0 76 25 255 255 255   # yellow
"""
