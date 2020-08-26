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
        """
        self.left_fit = None
        self.right_fit = None
        self.parameters = get_params()
        self.window_name = "Lane lines"
        self.debug = debug
        self.f = open("debug.txt", 'w')

    def __exit__(self):
        """Close all file streams."""
        self.f.close()

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        return self.fit_poly(img)

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

        # HYPERPARAMETERS
        # Number of sliding windows
        nwindows = self.parameters["nwindows"]
        # With of the the windows +/- margin
        margin = self.parameters["margin"]
        # Mininum number of pixels found to recenter window
        minpix = self.parameters["minpix"]

        # Height of of windows - based on nwindows and image shape
        window_height = np.int(img.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to reveice left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if self.debug:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                            (win_xleft_high, win_y_high), (0,255,0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                            (win_xright_high, win_y_high), (0,255,0), 2)

            good_left_inds = []
            good_right_inds = []

            for i in range(len(nonzeroy)):
                if (nonzeroy[i]>=win_y_low) and (nonzeroy[i]<=win_y_high):
                    if (nonzerox[i]>=win_xleft_low) and (nonzerox[i]<=win_xleft_high):
                        good_left_inds.append(i)

                    if (nonzerox[i]>=win_xright_low) and (nonzerox[i]<=win_xright_high):
                        good_right_inds.append(i)

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds).astype(int)
        right_lane_inds = np.concatenate(right_lane_inds).astype(int)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        print(self.left_fit, self.right_fit, file=self.f)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Visualization
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)

            if self.debug:
                cv2.circle(out_img, (l, y), 1, (0, 255, 0))
                cv2.circle(out_img, (r, y), 1, (0, 255, 0))
            else:
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return out_img

    def display(self, img):
        """ Display result of lane lines detection.

        Parameters:
            img (np.array): a binary warped image
        """
        self.img = img
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.forward(self.img))
