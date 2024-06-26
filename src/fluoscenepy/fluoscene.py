# -*- coding: utf-8 -*-
"""
Main script for the 'fluoscenepy' package.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""

# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Union
import random

# %% Local imports


# %% Scene (image) class def.
class UscopeScene():
    """
    Replicate the common fluorescences microscopic image (frame or 'scene').

    This class simulates the bright objects with round and elongate shapes as the basic examples of imaged objects. \n
    The ratio behind development of this class - to get the ground truth images for image processing workflows tests. \n
    For more complex and useful cases, please, check the References.

    References
    ----------
    [1]

    """

    # Class parameters
    width: int; height: int; __warn_message: str = ""; max_pixel_value = 255; img_type = np.uint8
    acceptable_img_types = ['uint8', 'uint16', 'float', np.uint8, np.uint16, np.float64]

    def __init__(self, width: int, height: int, img_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8'):
        """
        Initialize the class wrapper for store the current "scene" or microscopic image.

        Parameters
        ----------
        width : int
            Width of the scene in pixels.
        height : int
            Height of the scene in pixels.
        img_type : str | np.uint8 | np.uint16 | np.float64, optional
            Image type, for supported ones see the acceptable types. The default is 'uint8'.

        Raises
        ------
        ValueError
            If provided parameters are incosistent, e.g. width or height < 2.

        Returns
        -------
        None.

        """
        # Check provided width and height
        if width < 2 or height < 2:
            raise ValueError(f"Provided dimensions ({width}x{height}) is less than 2")
        if width > 10000 or height > 10000:
            __warn_message = f"Provided dimensions ({width}x{height}) pixels are unrealistic for the common image"
            warnings.warn(__warn_message)
        self.width = width; self.height = height
        # Check an image type
        if img_type not in self.acceptable_img_types:
            raise ValueError(f"Provided image type '{img_type}' not in the acceptable list of types: " + str(self.acceptable_img_types))
        else:
            if img_type == 'uint16' or img_type == np.uint16:
                self.max_pixel_value = 65535; self.img_type = np.uint16
            elif img_type == 'float' or img_type == np.float64:
                self.max_pixel_value = 1.0; self.img_type = np.float64
        # Initialize zero scene
        self.image = np.zeros(shape=(height, width), dtype=self.img_type)

    # %% Scene manipulation
    def show_scene(self, str_id: str = ""):
        """
        Show interactively the stored in the class scene (image) by plotting it using matplotlib.

        Parameters
        ----------
        str_id : str, optional
            Unique string id for plotting several plots. The default is "".

        Returns
        -------
        None.

        """
        plt.close('all'); h, w = self.image.shape
        height_width_ratio = h/w; default_image_size = 5.8
        if len(str_id) == 0:
            str_id = str(random.randint(1, 100))
        if not plt.isinteractive():
            plt.ion()
        plt.figure("UscopeScene_"+str_id, figsize=(default_image_size, default_image_size*height_width_ratio))
        plt.imshow(self.image, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.tight_layout()

    def clear_scene(self):
        """
        Reinitialize the scene to zero (plain dark background).

        Returns
        -------
        None.

        """
        self.image = np.zeros(shape=(self.height, self.width), dtype=self.img_type)


# %% Object class definition
class FluoObj():
    """
    Modelling the fluorescent bright object with the specified type.

    It can be used for embedding it to the 'UscopeScene' class for building up the microscopic image.

    """

    # Class parameters
    shape_type: str; border_type: str; __acceptable_shape_types: list = ['round', 'r', 'elongated', 'curved']

    def __init__(self, shape_type: str = 'round', border_type: str = 'precise'):
        if shape_type in self.__acceptable_shape_types:
            self.shape_type = shape_type
        else:
            raise ValueError(f"Provided shape type {shape_type} not in acceptable list {self.__acceptable_shape_types}")


# %% Some tests
if __name__ == "__main__":
    scene = UscopeScene(width=145, height=123); scene.show_scene()
