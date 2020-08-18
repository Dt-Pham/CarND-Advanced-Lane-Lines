import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob

'''
src_vertices = np.float32([(550, 460),   # top-left
                           (150, 720),   # bottom-left
                           (1200, 720),  # bottom-right
                           (770, 460)])  # top-right

dst_vertices = np.float32([(0, 0),
                           (0, 720),
                           (1200, 720),
                           (770, 460)]
'''

parameters = {
    "s_thresh_low" : 10,
    "s_thresh_high" : 300,
    "sx_thresh_low": 0,
    "sx_thresh_high": 255
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

# class ROI:
