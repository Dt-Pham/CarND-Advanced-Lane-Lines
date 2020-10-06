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
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class Application(tk.Frame):
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self, master = None):
        """ Init Application"""
        super().__init__(master)
        self.master = master
        self.create_widgets()
        self.grid()
        self.filepath = ""
        self.image = None
        self.debug = True
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines(debug=self.debug)
        self.vidcap = None
        self.frames = []
        self.current_frameid = -1

    def create_widgets(self):
        self.winfo_toplevel().title("Parameter Tuning")
        
        self.hi_there = tk.Button(self, text="Load image", command=self.load_image)
        self.hi_there.grid(row=1)

        self.textbox = tk.Entry(self)
        self.textbox.insert(0, 'test_images/project_video_frame_1048.jpg')
        self.textbox.grid(row=0)

        self.save = tk.Button(self, text="Save params", command=self.save_parameters)
        self.save.grid(row=2)
        self.load = tk.Button(self, text="Load params", command=self.load_parameters)
        self.load.grid(row=3)

        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.grid(row=5)

        self.process_image_button = tk.Button(self, text="Process image", command=self.process_image)
        self.process_image_button.grid(row=1, column=1)

        self.process_video_button = tk.Button(self, text="Process video", command=self.process_video)
        self.process_video_button.grid(row=1, column=2)

        self.frame_textbox = tk.Entry(self)
        self.frame_textbox.grid(row=0, column=1)

        self.get_frame_button = tk.Button(self, text="Get frame from video", command=self.get_frame)
        self.get_frame_button.grid(row=2, column=1)

        self.save_frame_button = tk.Button(self, text="Save frame from video", command=self.save_frame)
        self.save_frame_button.grid(row=3, column=1)

        self.debug_button = tk.Button(self, text="Debug ON", command=self.toggle_debug)
        self.debug_button.grid(row=1, column=3)

    def load_image(self):
        self.current_frameid = -1
        self.filepath = self.textbox.get()
        self.image = cv2.imread(self.filepath)

        img = self.image
        cv2.imshow("original", img)

        self.thresholding.display(img)
        img = self.thresholding.forward(img)

        self.transform.display(img)
        img = self.transform.forward(img)

        self.lanelines.display(img)

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.thresholding.forward(img)
        img2 = np.dstack((img, img, img))
        img = self.transform.forward(img)
        img = self.lanelines.forward(img)
        img3 = img
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.3, 0)
        out_img = self.lanelines.plot(out_img)
        if self.debug and self.current_frameid >= 0:
            cv2.putText(out_img, str(self.current_frameid), org=(700, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
            self.current_frameid += 1
        return out_img

    def process_image(self):
        self.current_frameid = -1
        img = cv2.imread(self.textbox.get())
        out_img = self.forward(img)
        cv2.imshow("output", out_img)

    def process_video(self):
        self.current_frameid = 0
        self.filepath = self.textbox.get()
        clip = VideoFileClip(self.filepath)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile("output_videos/"+self.filepath, audio=False)

    def get_frame(self):
        path = self.textbox.get()
        frame_id = int(self.frame_textbox.get())
        if (path != self.filepath) or (self.vidcap is None):
            self.vidcap = cv2.VideoCapture(path)
            self.filepath = path
            self.frames = []

        if len(self.frames) > frame_id:
            cv2.imshow("Frame {}".format(frame_id), self.frames[frame_id])
            return

        success, image = self.vidcap.read()
        while success:
            self.frames.append(image)
            if len(self.frames) > frame_id:
                cv2.imshow("Frame {}".format(frame_id), self.frames[frame_id])
                return
            success, image = self.vidcap.read()

    def save_frame(self):
        path = self.textbox.get()
        frame_id = int(self.frame_textbox.get())

        l = path.rfind('/')+1
        r = path.find('.')
        output_path = "test_images/"+path[l:r]+"_frame_{}.jpg".format(frame_id)
        self.get_frame()
        cv2.imwrite(output_path, self.frames[frame_id])

    def toggle_debug(self):
        self.debug = not self.debug
        if self.debug:
            self.debug_button.config(text="Debug ON")
        else:
            self.debug_button.config(text="Debug OFF")
        self.lanelines.debug = self.debug

    def save_parameters(self):
        self.thresholding.save_parameters()
        save_params()

    def load_parameters(self):
        load_params()
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines(debug=self.debug)

root = tk.Tk()
root.geometry("500x300")
app = Application(master=root)
app.mainloop()