import cv2
import numpy as np
from utils import *

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        window_name (str): Name of the window when display() is called
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self, debug=True):
        """Init Lanelines.

        Parameters:
            debug (boolean): If true, the display function will show more details
            parameters (dict): Dictionary that contains all parameters needed for the pipeline
            window_name (str): Name of the window when display() is called
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.debug = debug
        self.parameters = get_params()
        self.window_name = "Lane lines"
        self.debug_file = open("debug.txt", 'w')

        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = self.parameters["nwindows"]
        # Width of the the windows +/- margin
        self.margin = self.parameters["margin"]
        # Mininum number of pixels found to recenter window
        self.minpix = self.parameters["minpix"]

    def __exit__(self):
        """Close all file streams."""
        self.debug_file.close()

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window
        
        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def draw_window(self, out_img, center, margin, height, color=(0, 255, 0), thickness=2):
        """ Draw slicing window on the output image.

        Parameters:
            out_img (np.array): Output image
            center (tuple): coordinates of center of the rectangle
            margin (int): half width of the window
            height (int): height of the window
            color (tuple): color to draw window in RGB
            thickness (int): thickness of the window
        """
        margin = self.parameters["margin"]
        height = int(self.img.shape[0]//self.parameters["nwindows"])
        cv2.rectangle(out_img, (center[0]-margin, center[1]-height//2),
                      (center[0]+margin, center[1]+height//2), color, thickness)

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = np.int(img.shape[0]//self.nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def distance_from_curve(self, point, lane="left"):
        assert lane in ["left", "right"], "argument is not 'left' or 'right'"

        curve = self.left_fit if lane == "left" else self.right_fit
        x, y = point
        x0 = np.dot(curve, [y*y, y, 1])
        return abs(x0-x)

    def draw_curve(self, img, curve):
        for y in range(img.shape[1]):
            x = int(np.dot(curve, [y*y, y, 1]))
            cv2.circle(img, (x, y), 2, color=(0, 255, 0))

    def find_from_previous(self, img):
        """ Find lane pixels from a binary warped image using 
        curves from previous frame.

        Parameters:
            img (np.array): A binary warped image.
        
        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
        """
        assert len(img.shape) == 2

        # return self.find_lane_pixels(img)
        if (self.left_fit is None) or (self.right_fit is None):
            return self.find_lane_pixels(img)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        if self.debug:
            self.draw_curve(out_img, self.left_fit - np.array([0, 0, self.margin]))
            self.draw_curve(out_img, self.left_fit + np.array([0, 0, self.margin]))

            self.draw_curve(out_img, self.right_fit - np.array([0, 0, self.margin]))
            self.draw_curve(out_img, self.right_fit + np.array([0, 0, self.margin]))

        leftx, lefty, rightx, righty = [], [], [], []
        for i in range(len(self.nonzerox)):
            if self.distance_from_curve((self.nonzerox[i], self.nonzeroy[i]), "left") < self.margin:
                leftx.append(self.nonzerox[i])
                lefty.append(self.nonzeroy[i])

            if self.distance_from_curve((self.nonzerox[i], self.nonzeroy[i]), "right") < self.margin:
                rightx.append(self.nonzerox[i])
                righty.append(self.nonzeroy[i])

            # print(self.distance_from_curve((self.nonzerox[i], self.nonzeroy[i]), "left"))

        for y in range(680, 720):
            x0 = int(np.dot(self.left_fit, [y*y, y, 1]))
            x1 = int(np.dot(self.right_fit, [y*y, y, 1]))
            for i in range(-19, 20):
                leftx.append(x0+i)
                rightx.append(x1+i)
                lefty.append(y)
                righty.append(y)

        return leftx, lefty, rightx, righty, out_img

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            if self.debug:
                # Draw the windows on the visualization image
                self.draw_window(out_img, center_left, self.margin, self.window_height)
                self.draw_window(out_img, center_right, self.margin, self.window_height)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """

        leftx, lefty, rightx, righty, out_img = self.find_from_previous(img)

        if len(lefty) > 50:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 50:
            self.right_fit = np.polyfit(righty, rightx, 2)
        # print(self.left_fit, self.right_fit, file=self.debug_file)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)

            if self.debug:
                cv2.circle(out_img, (l, y), 1, (0, 255, 0))
                cv2.circle(out_img, (r, y), 1, (0, 255, 0))
            else:
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        if len(lefty) > 0 : out_img[lefty, leftx] = [255, 0, 0]
        if len(righty) > 0: out_img[righty, rightx] = [0, 0, 255]

        return out_img

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        cv2.putText(out_img, str(self.left_fit), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(out_img, str(self.right_fit), org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        return out_img

    def display(self, img):
        """ Display result of lane lines detection.

        Parameters:
            img (np.array): a binary warped image
        """
        self.img = img
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.forward(self.img))
