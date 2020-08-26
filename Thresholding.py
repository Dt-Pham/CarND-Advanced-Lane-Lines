import cv2
from utils import *

class Thresholding:
    def __init__(self):
        self.parameters = get_params()
        self.s_thresh = (self.parameters["s_thresh_low"], self.parameters["s_thresh_high"])
        self.sx_thresh = (self.parameters["sx_thresh_low"], self.parameters["sx_thresh_high"])
        self.img = None
    
    def forward(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobel = np.abs(sobelx)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        # Threshold x gradient
        sxbinary = np.uint8((scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1]))

        # Threshold color channel
        s_binary = np.uint8((s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1]))
        
        # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        binary = (s_binary | sxbinary) * 255
        return binary

    
    def display(self, img):
        """ Similar to forward function but display image as well
        """
        self.img = img
        cv2.namedWindow("thresholded")
        self.create_trackbar()
        cv2.imshow("thresholded", self.forward(self.img))

    def create_trackbar(self):
        cv2.createTrackbar("S_low", "thresholded", self.s_thresh[0], 360, lambda v: self.trackbar_callback("s_thresh_low", v))
        cv2.createTrackbar("S_high", "thresholded", self.s_thresh[1], 360, lambda v: self.trackbar_callback("s_thresh_high", v))
        cv2.createTrackbar("Sx_low", "thresholded", self.sx_thresh[0], 255, lambda v: self.trackbar_callback("sx_thresh_low", v))
        cv2.createTrackbar("Sx_high", "thresholded", self.sx_thresh[1], 255, lambda v: self.trackbar_callback("sx_thresh_high", v))

    def trackbar_callback(self, param, value):
        self.parameters[param] = value
        set_params(self.parameters)
        self.s_thresh = (self.parameters["s_thresh_low"], self.parameters["s_thresh_high"])
        self.sx_thresh = (self.parameters["sx_thresh_low"], self.parameters["sx_thresh_high"])
        cv2.imshow("thresholded", self.forward(self.img))

