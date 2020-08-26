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
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.create_widgets()
        self.grid()
        self.filepath = ""
        self.image = None
        self.debug = True
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines(debug=self.debug)

        self.vidcap = None
        self.frames = []

    def create_widgets(self):
        self.winfo_toplevel().title("Parameter Tuning")
        
        self.hi_there = tk.Button(self, text="Load image", command=self.load_image)
        self.hi_there.grid(row=1)

        self.textbox = tk.Entry(self)
        self.textbox.insert(0, 'test_images/test1.jpg')
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

    def load_image(self):
        self.debug = True
        self.filepath = self.textbox.get()
        self.image = cv2.imread(self.filepath)

        thresholding = Thresholding()
        transform = PerspectiveTransformation()
        lanelines = LaneLines()

        img = self.image
        cv2.imshow("original", img)

        thresholding.display(img)
        img = thresholding.forward(img)

        transform.display(img)
        img = transform.forward(img)

        lanelines.display(img)

    def forward(self, img):
        out_img = np.copy(img)
        img = self.thresholding.forward(img)
        img = self.transform.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.3, 0)
        return out_img
    
    def process_image(self):
        self.debug = False
        img = cv2.imread(self.textbox.get())
        out_img = self.forward(img)
        cv2.imshow("output", out_img)

    def process_video(self):
        self.debug = False
        self.filepath = self.textbox.get()
        clip = VideoFileClip(self.filepath)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile("output_videos/"+path, audio=False)

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

        return

    def save_parameters(self):
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