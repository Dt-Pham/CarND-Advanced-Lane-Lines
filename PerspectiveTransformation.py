import cv2
from utils import *

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        parameters (dict): Dictionary containing coordinates of source and destination vertices
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        name (str): Name of the window when display method is called
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        """Init PerspectiveTransformation."""
        self.parameters = get_params()
        self.src = self.parameters["src_vertices"]
        self.dst = self.parameters["dst_vertices"]
        self.name = "Perspective transformation"
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            img_size (tuple): Size of the image (width, height)
            flags : flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)

    def display(self, img):
        """ Display top view of a front view image

        Parameters:
            img (np.array): A front view image
        """
        self.img = img
        cv2.namedWindow(self.name)
        cv2.imshow(self.name, self.forward(self.img))
