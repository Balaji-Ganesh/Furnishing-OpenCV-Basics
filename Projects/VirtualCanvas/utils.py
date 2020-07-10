# Import the required libararies..
import os
import cv2
import numpy as np

# Create dictionary of the color-values as key and BGR codes as value
colors_BGR = {'Orange': (91, 153, 255),
              'Blue': (255, 0, 0),
              'Yellow': (56, 252, 255),
              'Red': (0, 0, 255),
              'Green': (0, 255, 0),
              'SkyBlue': (255, 247, 0),
              'Pink': (255, 0, 243),
              'Black': (0, 0, 0),
              'Violet': (255, 117, 116)
              }  # Don't forget to change below at color values (in take_new_data() function..) if made any changes to the dictionary..


# Required functions..
def load_data(num_markers, debug_mode=False):
    """
    This function will load the data from this current directory if already present, else takes new data and returns to the calling function.
    :param num_markers: No. of markers the user need the HSV and BGR values for..
    :param debug_mode: Displays interactive messages if set to True, else not. Default is False
    :return: Desired no. of marker's HSV values and respective BGR format colors (as list format..)
    """
    # Required variables
    # markers_HSV = np.array(dtype=np.uint8)
    # marker_colors = np.array(dtype=np.uint8)
    # Get the path of the directory currently in which we are working..
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Now search for the NUMPY file (.npy) file..
    for root, dirs, files in os.walk(dir_path):
        # If the file is present, then load the file data and send it to the called function..
        if 'markers_HSV.npy' in files and 'marker_colors.npy' in files:
            if debug_mode: print("Files present, going for validation of no. of colors..")
            # Load the data from the files.. and converted into list type from numpy type, to just keep simple(ie., every where same type of object returned to the called function..).. if data in the loaded data satisfies with the desired num of markers..
            markers_HSV = np.load('markers_HSV.npy').reshape((3, 2, 3))   # Reshaping, as getting the shape as (3, 1, 2, 3) when loaded ...
            marker_colors = np.load('marker_colors.npy')

            # Validate the data, whether the loaded data contains values of the desired markers..
            if num_markers == markers_HSV.shape[0]:  # As equal as num_markers == markers_colors.shape[0]
                if debug_mode: print("Previous data satisfied with desired no. of marker colors.. returning the previous data results...")
                # return list(markers_HSV), list(marker_colors)  # Converting these into lists as just said above 5 lines
                #print("marker_colors: ", list(marker_colors[0]))
                return markers_HSV.tolist(), marker_colors.tolist()
            # Take the new data that equals to the num_markers.
            else:
                if debug_mode: print("utils.py: Data present but, not doesn't meet the requirement of no. of markers.\n"
                                     "So Taking the data freshly..")
                markers_HSV, marker_colors = take_new_data(num_markers=num_markers)

                # Save the data..
                np.save(file="markers_HSV.npy", arr=markers_HSV)
                np.save(file="marker_colors.npy", arr=marker_colors)
        # (Even)if not present, then take the new data..
        else:
            if debug_mode: print("Sorry, Files not present, taking the new data..")
            markers_HSV, marker_colors = take_new_data(num_markers=num_markers)
            # Save the data..
            np.save(file="markers_HSV.npy", arr=markers_HSV)
            np.save(file="marker_colors.npy", arr=marker_colors)

        return markers_HSV, marker_colors


def take_new_data(num_markers):
    """
    This function will take the new data of HSV and respective color codes.
    :return: HSV color codes and its respective colors (as BGR format) as lists
    """
    # Create empty lists to store the values...
    markers_HSV = []
    markers_colors = []
    for i in range(num_markers):
        color = ""
        # get the proper valid color
        while color not in colors_BGR.keys():
            print("Please Choose a color from the list: (case-sensitive..)\nOrange, Blue, Yellow, Red, Green, SkyBlue, Pink, Black, 'Violet'\nIf needed more colors, please add them to the list..!!")
            color = input("Color : ")
        print("Adjust the HSV values in the windows that appear..!!")
        markers_colors.append(colors_BGR[color])  # Get the color code and append it to the list.. (for Drawing on canvas)
        markers_HSV.append(find_markers_HSV())  # Get the HSV values (For masking purpose)
    return markers_HSV, markers_colors


def dummy(dummy):
    """
    This function is called - When track bar's value is changed.
    Currently we are not performing any action with this function, named dummy to resemble the theme of use.
    :param dummy:
    :return:
    """
    pass


def find_markers_HSV():
    """
    This function helps in finding the HSV values for the specific marker.
    This function is set to ran for 3 (or no. of markers chosen) times.
    Each time specific marker's HSV values must be adjusted, and saved automatically in the list at every quit of window(by Esc key).
    :return: the list of two lists -> one the HSV min values and the other with HSV max values.
    """
    # required variables..
    markers_HSV = []  # An empty list to store the min and max HSV values..

    # Create the track bars..
    cv2.namedWindow(winname="Filter the markers..")
    cv2.resizeWindow(winname="Filter the markers..", width=500, height=400)
    cv2.createTrackbar("HUE min", "Filter the markers..", 0, 255, dummy)
    cv2.createTrackbar("HUE max", "Filter the markers..", 255, 255, dummy)
    cv2.createTrackbar("SAT min", "Filter the markers..", 0, 255, dummy)
    cv2.createTrackbar("SAT max", "Filter the markers..", 255, 255, dummy)
    cv2.createTrackbar("VAL min", "Filter the markers..", 0, 255, dummy)
    cv2.createTrackbar("VAL max", "Filter the markers..", 255, 255, dummy)

    # Get the cam feed..
    capture = cv2.VideoCapture(0)
    # Adjust the camera capture properties..
    capture.set(cv2.CAP_PROP_BRIGHTNESS, 200)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    while True:
        # Get the image..
        read_status, image = capture.read()
        # Convert to HSV..
        HSVimg = cv2.cvtColor(src=image.copy(), code=cv2.COLOR_BGR2HSV)
        # get the values from the track bar..
        hue_min = cv2.getTrackbarPos(trackbarname="HUE min", winname="Filter the markers..")
        hue_max = cv2.getTrackbarPos(trackbarname="HUE max", winname="Filter the markers..")
        sat_min = cv2.getTrackbarPos(trackbarname="SAT min", winname="Filter the markers..")
        sat_max = cv2.getTrackbarPos(trackbarname="SAT max", winname="Filter the markers..")
        val_min = cv2.getTrackbarPos(trackbarname="VAL min", winname="Filter the markers..")
        val_max = cv2.getTrackbarPos(trackbarname="VAL max", winname="Filter the markers..")

        # Group the lower and higher boundary HSV values..
        lower_boundary = np.array([hue_min, sat_min, val_min])
        upper_boundary = np.array([hue_max, sat_max, val_max])

        # Now filter the image..
        maskImg = cv2.inRange(src=HSVimg, lowerb=lower_boundary, upperb=upper_boundary)
        cv2.imshow("Mask Image", maskImg)

        # Show the filter image by bitwise_and the original image and maskImg
        filter_img = cv2.bitwise_and(src1=image, src2=image, mask=maskImg)
        cv2.imshow("Filtered Image", filter_img)

        # While Quitting the windows(by Esc key), save the trackbar values
        if cv2.waitKey(1) == 27:
            markers_HSV.append([[hue_min, sat_min, val_min], [hue_max, sat_max, val_max]])
            print("Final HSV values as: HUE_min, SAT_min, VAL_min, HUE_max, SAT_max, VAL_max", hue_min, sat_min, val_min, hue_max, sat_max, val_max)
            cv2.destroyAllWindows()
            break

    return markers_HSV


if __name__ == '__main__':
    load_data(num_markers=3, debug_mode=True)
    print("Data loaded successfully..")