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
from pathlib import Path
from matplotlib.patches import Circle

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from utils.raw_objects_gen import continuous_shaped_bead
else:
    from .utils.raw_objects_gen import continuous_shaped_bead


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
            Unique string id for plotting several plots with unique Figure() names. The default is "".

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
class FluorObj():
    """
    Modelling the fluorescent bright object with the specified type.

    It can be used for embedding it to the 'UscopeScene' class for building up the microscopic image.

    """

    # Class parameters
    shape_type: str = ""; border_type: str = ""; shape_method: str = ""; explicit_shape_name: str = ""; profile: np.ndarray = None
    __acceptable_shape_types: list = ['round', 'r', 'elongated', 'el', 'curved', 'c']  # shape types of the object
    __acceptable_border_types: list = ['precise', 'pr', 'computed', 'co']; radius: float = 0.0; a: float = 0.0; b: float = 0.0
    typical_sizes: tuple = ()  # for storing descriptive parameters for curve describing the shape of the object
    # below - storing names of implemented computing functions for the define continuous shape
    __acceptable_shape_methods = ['gaussian', 'g', 'lorentzian', 'lor', 'derivative of logistic func.', 'dlogf', 'bump square', 'bump2']

    def __init__(self, typical_size: Union[float, int, tuple], center_shifts: tuple = (0.0, 0.0), shape_type: str = 'round',
                 border_type: str = 'precise', shape_method: str = ''):
        # Sanity checks of the input values
        if shape_type in self.__acceptable_shape_types:
            self.shape_type = shape_type
        else:
            raise ValueError(f"Provided shape type '{shape_type}' not in acceptable list {self.__acceptable_shape_types}")
        if border_type in self.__acceptable_border_types:
            self.border_type = border_type
        else:
            raise ValueError(f"Provided border type '{border_type}' not in acceptable list {self.__acceptable_border_types}")
        if self.shape_type == "round" or self.shape_type == "r":
            if isinstance(typical_size, float):
                if typical_size < 0.5:
                    raise ValueError(f"Expected typical size (radius) should be larger than 0.5px, provided: {typical_size}")
                else:
                    self.radius = typical_size
            elif isinstance(typical_size, int):
                typical_size = float(typical_size)
                if typical_size < 0.5:
                    raise ValueError(f"Expected typical size (radius) should be larger than 0.5px, provided: {typical_size}")
                else:
                    self.radius = typical_size
            else:
                raise ValueError(f"For round particle expected type of typical size is float or int, not {type(typical_size)}")
        else:
            if self.shape_type == "elongated" or self.shape_type == "el":
                if len(typical_size) != 2:
                    raise ValueError("For elongated particle expected length of typical size tuple is equal 2")
                else:
                    a, b = typical_size; max_size = max(a, b); min_size = min(a, b)
                    if max_size < 0.5 or min_size < 0.0:
                        raise ValueError("Expected sizes a, b should be positive and more than 0.5px")
            else:
                if len(typical_size) <= 2:
                    raise ValueError("For curved particle expected length of typical size tuple is more than 2")
                else:
                    self.typical_sizes = typical_size
        if self.border_type == "computed" or self.border_type == "co":
            if shape_method in self.__acceptable_shape_methods:
                self.shape_method = shape_method
                if self.shape_method == 'g':
                    self.explicit_shape_name = 'gaussian'
                elif self.shape_method == 'lor':
                    self.explicit_shape_name = 'lorentzian'
                elif self.shape_method == 'dlogf':
                    self.explicit_shape_name = 'derivative of logistic func.'
                elif self.shape_method == 'bump2':
                    self.explicit_shape_name = 'bump square'
            else:
                raise ValueError(f"Provided shape computation method '{shape_method}' not in acceptable list {self.__acceptable_shape_methods}")
        # Assuming that center shifts provided as the tuple with x shift, y shift floats in pixels
        if len(center_shifts) == 2:
            x_shift, y_shift = center_shifts
            if abs(x_shift) < 1.0 and abs(y_shift) < 1.0:
                self.center_shifts = center_shifts
            else:
                raise ValueError(f"One of the shifts '{center_shifts}' in pixels are more than 1px, "
                                 + "but the shifts are expected to be in the subpixel range")

    def get_shape(self):
        if (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "computed" or self.border_type == "co"):
            self.profile = continuous_shaped_bead(self.radius, self.center_shifts, bead_type=self.shape_method)

    def plot_shape(self, str_id: str = ""):
        """
        Plot interactively the profile of the object computed by get_shape() method along with the border of the object (for round objects).

        Parameters
        ----------
        str_id : str, optional
            Unique string id for plotting several plots with unique Figure() names. The default is "".

        Returns
        -------
        None.

        """
        if not plt.isinteractive():
            plt.ion()
        if self.profile is not None:
            plt.figure(f"Shape with parameters: {self.shape_type}, {self.border_type}, {self.explicit_shape_name}, {str_id}")
            axes_img = plt.imshow(self.profile, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.colorbar()
            plt.tight_layout()
            if self.shape_type == "round" or self.shape_type == "r":
                m_center, n_center = self.profile.shape; m_center = m_center // 2 + self.center_shifts[1]
                n_center = n_center // 2 + self.center_shifts[0]
                axes_img.axes.add_patch(Circle((n_center, m_center), self.radius, edgecolor='red', linewidth=1.5, facecolor='none'))


# %% Some tests
if __name__ == "__main__":
    plt.close("all")
    # scene = UscopeScene(width=145, height=123); scene.show_scene()
    gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g'); gb1.get_shape(); gb1.plot_shape()
    gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor'); gb2.get_shape(); gb2.plot_shape()
    gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf'); gb3.get_shape(); gb3.plot_shape()
    gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2'); gb4.get_shape(); gb4.plot_shape()
