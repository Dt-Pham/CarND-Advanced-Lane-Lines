import tkinter as tk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from utils import *

def pipeline(img, s_thresh=(10, 300), sx_thresh=(0, 255)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.abs(sobelx)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Threshold x gradient
    sxbinary = np.uint8((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))

    # Threshold color channel
    s_binary = np.uint8((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))
    
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    binary = (s_binary | sxbinary) *255
    return binary

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

class Application(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.create_widgets()
        self.grid()
        self.filepath = ""
        self.images = {}

    def create_widgets(self):
        self.hi_there = tk.Button(self, text="Load image", command=self.load_image)
        self.hi_there.grid(row=1)

        self.textbox = tk.Entry(self)
        self.textbox.insert(0, 'test_images/test1.jpg')
        self.textbox.grid(row=0)
        
        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.grid(row=2)
    
    def load_image(self):
        self.filepath = self.textbox.get()
        self.images["original"] = cv2.imread(self.filepath)
        self.images["thresholded"] = pipeline(self.images["original"])

        cv2.imshow("original", self.images["original"])
        cv2.setMouseCallback("original", click_event)
        
        cv2.namedWindow("thresholded")
        self.create_trackbar()
        cv2.imshow("thresholded", self.images["thresholded"])

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    def create_trackbar(self):
        cv2.createTrackbar("S_low", "thresholded", 0, 360, lambda v: self.trackbar_callback("s_thresh_low", v))
        cv2.createTrackbar("S_high", "thresholded", 0, 360, lambda v: self.trackbar_callback("s_thresh_high", v))
        cv2.createTrackbar("Sx_low", "thresholded", 0, 255, lambda v: self.trackbar_callback("sx_thresh_low", v))
        cv2.createTrackbar("Sx_high", "thresholded", 0, 255, lambda v: self.trackbar_callback("sx_thresh_high", v))


    def trackbar_callback(self, param, value):
        parameters[param] = value
        s_thresh = (parameters["s_thresh_low"], parameters["s_thresh_high"])
        sx_thresh = (parameters["sx_thresh_low"], parameters["sx_thresh_high"])
        self.images["thresholded"] = pipeline(self.images["original"], s_thresh, sx_thresh)
        cv2.imshow("thresholded", self.images["thresholded"])

root = tk.Tk()
root.geometry("500x300")
app = Application(master=root)
app.mainloop()