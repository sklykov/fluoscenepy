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
    from utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead
else:
    from .utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead


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
    __acceptable_shape_methods = ['gaussian', 'g', 'lorentzian', 'lor', 'derivative of logistic func.', 'dlogf', 'bump square', 'bump2',
                                  'bump cube', 'bump3', 'bump ^8', 'bump8', 'smooth circle', 'smcir']
    center_shifts: tuple = (0.0, 0.0)

    def __init__(self, typical_size: Union[float, int, tuple], center_shifts: tuple = (0.0, 0.0), shape_type: str = 'round',
                 border_type: str = 'precise', shape_method: str = ''):
        f"""
        Initialize the class representation of a fluorescent object.

        The difference between used parameters can be observed by plotting of the calculated shapes (profiles) by get_shape() method.

        Parameters
        ----------
        typical_size : Union[float, int, tuple]
            Typical sizes of the object, e.g. for a bead - radius (float or int), for ellipsoid - tuple with axis (a, b).
        center_shifts : tuple, optional
            Shifts in pixels of the object center, should be less than 1px. The default is (0.0, 0.0).
        shape_type : str, optional
            Supporeted shape types: {self.__acceptable_shape_types}. \n
            Currently implemented: 'round' or 'r' - for the circular bead object. The default is 'round'.
        border_type : str, optional
            Type of intensity of the border pixels calculation. Supported border types: 'precise', 'pr', 'computed', 'co'. \n
            The 'computed' or 'co' type should be accomponied with the specification of the shape method parameter. \n
            The 'precise' or 'pr' type corresponds to the developed counting area of a pixel laying within the border (e.g., circular) of an object. \n
            The default is 'precise'.
        shape_method : str, optional
            Shape method calculation, supported ones: {self.__acceptable_shape_methods}. The default is ''.

        Raises
        ------
        ValueError
            See the provided error description for details.

        Returns
        -------
        None.

        """
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
            typical_size = float(typical_size)  # assuming that the input parameter can be converted to the float type
            if typical_size < 0.5:
                raise ValueError(f"Expected typical size (radius) should be larger than 0.5px, provided: {typical_size}")
            else:
                self.radius = typical_size
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
                elif self.shape_method == 'bump3':
                    self.explicit_shape_name = 'bump cube'
                elif self.shape_method == 'bump8':
                    self.explicit_shape_name = 'bump ^8'
                elif self.shape_method == 'smcir':
                    self.explicit_shape_name = 'smooth circle'
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

    def get_shape(self) -> np.ndarray:
        """
        Calculate and return 2D intensity distribution of the object shape.

        Raises
        ------
        NotImplementedError
            For some set of allowed parameters the calculation hasn't been yet implemented.

        Returns
        -------
        2D shape of the object (intensity representation).

        """
        if (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "computed" or self.border_type == "co"):
            self.profile = continuous_shaped_bead(self.radius, self.center_shifts, bead_type=self.shape_method)
        elif (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "precise" or self.border_type == "pr"):
            self.profile = discrete_shaped_bead(self.radius, self.center_shifts)
        else:
            raise NotImplementedError("This set of input parameters hasn't yet been implemented")

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
            plt.figure(f"Shape with parameters: {self.shape_type}, {self.border_type}, {self.explicit_shape_name}, "
                       + f"center: {self.center_shifts} {str_id}")
            axes_img = plt.imshow(self.profile, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.colorbar(); plt.tight_layout()
            if self.shape_type == "round" or self.shape_type == "r":
                m_center, n_center = self.profile.shape
                if self.center_shifts[0] >= 0.0:
                    m_center = m_center // 2 + self.center_shifts[0]
                else:
                    m_center = m_center // 2 + self.center_shifts[0] + 1.0
                if self.center_shifts[1] >= 0.0:
                    n_center = n_center // 2 + self.center_shifts[1]
                else:
                    n_center = n_center // 2 + self.center_shifts[1] + 1.0
                axes_img.axes.add_patch(Circle((m_center, n_center), self.radius, edgecolor='red', linewidth=1.5, facecolor='none'))
                axes_img.axes.plot(m_center, n_center, marker='.', linewidth=3, color='red')


# %% Some tests
if __name__ == "__main__":
    plt.close("all"); test_computed_centered_beads = False; test_precise_centered_bead = False; test_computed_shifted_beads = True
    test_presice_shifted_beads = True; shifts = (-0.69, 0.44)

    # Testing the scene generation with a few objects
    # scene = UscopeScene(width=145, height=123); scene.show_scene()

    # Testing the centered round objects generation
    if test_computed_centered_beads:
        gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g'); gb1.get_shape(); gb1.plot_shape()
        gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor'); gb2.get_shape(); gb2.plot_shape()
        gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf'); gb3.get_shape(); gb3.plot_shape()
        gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2'); gb4.get_shape(); gb4.plot_shape()
        gb5 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump3'); gb5.get_shape(); gb5.plot_shape()
        gb6 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump8'); gb6.get_shape(); gb6.plot_shape()
        gb7 = FluorObj(typical_size=2.0, border_type='co', shape_method='smcir'); gb7.get_shape(); gb7.plot_shape()
    if test_precise_centered_bead:
        gb8 = FluorObj(typical_size=2.0); gb8.get_shape(); gb8.plot_shape()
    # Testing the shifted from the center objects generation
    if test_computed_shifted_beads:
        gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g', center_shifts=shifts); gb1.get_shape(); gb1.plot_shape()
        gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor', center_shifts=shifts); gb2.get_shape(); gb2.plot_shape()
        gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf', center_shifts=shifts); gb3.get_shape(); gb3.plot_shape()
        gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2', center_shifts=shifts); gb4.get_shape(); gb4.plot_shape()
        gb5 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump3', center_shifts=shifts); gb5.get_shape(); gb5.plot_shape()
        gb6 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump8', center_shifts=shifts); gb6.get_shape(); gb6.plot_shape()
        gb7 = FluorObj(typical_size=2.0, border_type='co', shape_method='smcir', center_shifts=shifts); gb7.get_shape(); gb7.plot_shape()
    if test_presice_shifted_beads:
        gb8 = FluorObj(typical_size=2.0, center_shifts=shifts); gb8.get_shape(); gb8.plot_shape()
