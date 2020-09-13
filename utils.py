import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pickle

_parameters = {
    "s_thresh_low" : 10,
    "s_thresh_high" : 300,
    "sx_thresh_low": 0,
    "sx_thresh_high": 255,
    "region_of_interest": np.ones((720, 1280), dtype=np.bool),
    "src_vertices": np.float32([(550, 460),    # top-left
                                (150, 720),    # bottom-left
                                (1200, 720),   # bottom-right
                                (770, 460)]),  # top-right
    "dst_vertices": np.float32([(100, 0),
                                (100, 720),
                                (1100, 720),
                                (1100, 0)]),
    "nwindows": 9,
    "margin": 100,
    "minpix": 50
}

def display_images(imgs):
    """
    Display list of images. 2 per row.
    """
    n = len(imgs)
    f, axes = plt.subplots(n//2, 2, figsize=(24, 9*((n+1)//2)), squeeze=False)
    
    for i, img in enumerate(imgs):
        if len(img.shape)==2:
            axes[i//2][i%2].imshow(img, cmap='gray')
        else:
            axes[i//2][i%2].imshow(img)


def display_polynomial(img, fit):
    """
    Plot polynomial on top of a image
    """
    out_img = np.copy(img)
    
    y = np.linspace(0, img.shape[0]-1, img.shape[0])

    pass

def get_params():
    return _parameters

def set_params(params):
    _parameters = params

def save_params():
    with open("params.pkl", "wb") as f:
        pickle.dump(_parameters, f)
    print("Saved parameters to params.pkl successfully")

def load_params():
    with open("params.pkl", "rb") as f:
        global _parameters
        _parameters = pickle.load(f)
    print("Load parameters successfully")
    print(_parameters)