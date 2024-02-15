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

# %% Local imports


# %% Main class def.
class UscopeScene():
    """Replicate the microscopic image (frame or scene)."""

    # Class parameters
    width: int; height: int; __warn_message = ""; max_pixel_value = 255; img_type = np.uint8
    acceptable_img_types = ['uint8', 'uint16', 'float', np.uint8, np.uint16, np.float64]

    def __init__(self, width: int, height: int, img_type='uint8'):
        # Check provided width and height
        if width < 2 or height < 2:
            raise ValueError(f"Provided dimensions ({width}x{height}) is less than 2")
        if width > 5000 or height > 5000:
            __warn_message = f"Provided dimensions ({width}x{height}) are unrealistic for the image"
            warnings.warn(__warn_message)
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

    def show_scene(self):
        """
        Show interactively the stored in the class scene (image).

        Returns
        -------
        None.

        """
        plt.close('all')
        if not plt.isinteractive():
            plt.ion()
        plt.figure("UscopeScene"); plt.imshow(self.image, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()


# %% Some tests
if __name__ == "__main__":
    scene = UscopeScene(width=145, height=123); scene.show_scene()
