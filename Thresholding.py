import cv2
from utils import *

MOUSE_SIZE = 10

class Thresholding:
    """ This class is for extracting relevant pixels in an image.

    Attributes:
        parameters (dict): Dictionary contains all parameters needed for the pipeline
        s_thresh (tuple): Low and high value of S channel in HLS color space
        sx_thresh (tuple): Low and high value of change in x direction
        window_name (str): Name of the window when display() is called
        img (np.array): Image that is being processed
    """
    def __init__(self):
        """ Init Thresholding."""
        self.parameters = get_params()
        self.s_thresh = (self.parameters["s_thresh_low"], self.parameters["s_thresh_high"])
        self.sx_thresh = (self.parameters["sx_thresh_low"], self.parameters["sx_thresh_high"])
        self.roi = self.parameters["region_of_interest"]
        self.window_name = "thresholded"
        self.img = None

        self._left_clicked = False
        self._right_clicked = False

    def houghlines(self, gray):
        """ Take an image and find lines in it."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        cv2.imshow("Canny", edges)
        return edges

    def forward(self, img):
        """ Take an image and extract all relavant pixels.

        Parameters:
            img (np.array): An image

        Returns:
            binary (np.array): A binary image
        """
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        cv2.imshow("S channel", s_channel)
        cv2.imshow("Region of interest", (np.uint8(self.roi) * 255))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=5)

        gray = self.houghlines(gray)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.abs(sobelx)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Threshold x gradient
        sxbinary = np.uint8((scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1]))

        # Threshold color channel
        s_binary = np.uint8((s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1]))
        # print(s_binary.shape)
        # print(self.roi.shape)
        binary = ((s_binary | sxbinary) & self.roi) * 255
        return binary

    def display(self, img):
        """ Display a binary image represents all relevant pixels.

        Paramters:
            img (np.array): An image
        """
        self.img = img
        cv2.namedWindow(self.window_name)
        self.create_trackbar()
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.forward(self.img))

    def create_trackbar(self):
        """Create trackbar to tune parameters."""
        cv2.createTrackbar("S_low", self.window_name, self.s_thresh[0], 360, lambda v: self.trackbar_callback("s_thresh_low", v))
        cv2.createTrackbar("S_high", self.window_name, self.s_thresh[1], 360, lambda v: self.trackbar_callback("s_thresh_high", v))
        cv2.createTrackbar("Sx_low", self.window_name, self.sx_thresh[0], 255, lambda v: self.trackbar_callback("sx_thresh_low", v))
        cv2.createTrackbar("Sx_high", self.window_name, self.sx_thresh[1], 255, lambda v: self.trackbar_callback("sx_thresh_high", v))

    def trackbar_callback(self, param, value):
        """ Set parameters whenever value of a trackbar changes.

        Parameters:
            param (str): Name of the parameter
            value (int): Value of the parameter
        """
        self.parameters[param] = value
        set_params(self.parameters)
        self.s_thresh = (self.parameters["s_thresh_low"], self.parameters["s_thresh_high"])
        self.sx_thresh = (self.parameters["sx_thresh_low"], self.parameters["sx_thresh_high"])
        cv2.imshow(self.window_name, self.forward(self.img))

    def save_parameters(self):
        set_params(self.parameters)

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.roi = np.ones((720, 1280), dtype=np.bool)
            print("set ROI to all 1s")
            cv2.imshow(self.window_name, self.forward(self.img))
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self.roi = np.zeros((720, 1280), dtype=np.bool)
            print("set ROI to all 0s")
            cv2.imshow(self.window_name, self.forward(self.img))
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._left_clicked = True
            x0 = max(0, x-MOUSE_SIZE)
            x1 = min(1280, x+MOUSE_SIZE)
            y0 = max(0, y-MOUSE_SIZE)
            y1 = min(720, y+MOUSE_SIZE)
            self.roi[y0:y1, x0:x1] = True
            cv2.imshow(self.window_name, self.forward(self.img))
            print(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._right_clicked = True
            x0 = max(0, x-MOUSE_SIZE)
            x1 = min(1280, x+MOUSE_SIZE)
            y0 = max(0, y-MOUSE_SIZE)
            y1 = min(720, y+MOUSE_SIZE)
            self.roi[y0:y1, x0:x1] = False
            print(x, y)
            cv2.imshow(self.window_name, self.forward(self.img))
        elif event == cv2.EVENT_LBUTTONUP:
            self._left_clicked = False
        elif event == cv2.EVENT_RBUTTONUP:
            self._right_clicked = False
        elif event == cv2.EVENT_MOUSEMOVE and self._left_clicked:
            x0 = max(0, x-MOUSE_SIZE)
            x1 = min(1280, x+MOUSE_SIZE)
            y0 = max(0, y-MOUSE_SIZE)
            y1 = min(720, y+MOUSE_SIZE)
            self.roi[y0:y1, x0:x1] = True
            cv2.imshow(self.window_name, self.forward(self.img))
            print(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._right_clicked:
            x0 = max(0, x-MOUSE_SIZE)
            x1 = min(1280, x+MOUSE_SIZE)
            y0 = max(0, y-MOUSE_SIZE)
            y1 = min(720, y+MOUSE_SIZE)
            self.roi[y0:y1, x0:x1] = False
            cv2.imshow(self.window_name, self.forward(self.img))
            print(x, y)