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
from matplotlib.patches import Circle, Ellipse

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead, discrete_shaped_ellipse
else:
    from .utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead, discrete_shaped_ellipse


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
    width: int = 4; height: int = 4; __warn_message: str = ""; max_pixel_value = 255; img_type = np.uint8
    acceptable_img_types: list = ['uint8', 'uint16', 'float', np.uint8, np.uint16, np.float64]
    max_pixel_value_uint8: int = 255; max_pixel_value_uint16: int = 65535

    def __init__(self, width: int, height: int, image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8'):
        """
        Initialize the class wrapper for store the current "scene" or microscopic image.

        Parameters
        ----------
        width : int
            Width of the scene in pixels.
        height : int
            Height of the scene in pixels.
        image_type : str | np.uint8 | np.uint16 | np.float64, optional
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
        if width < 3 or height < 3:
            raise ValueError(f"Provided dimensions ({width}x{height}) is less than 3")
        if width > 10000 or height > 10000:
            __warn_message = f"Provided dimensions ({width}x{height}) pixels are unrealistic for the common image"
            warnings.warn(__warn_message)
        self.width = width; self.height = height
        # Check an image type
        if image_type not in self.acceptable_img_types:
            raise ValueError(f"Provided image type '{image_type}' not in the acceptable list of types: " + str(self.acceptable_img_types))
        else:
            if image_type == 'uint16' or image_type is np.uint16:
                self.max_pixel_value = self.max_pixel_value_uint16; self.img_type = np.uint16
            elif image_type == 'float' or image_type is np.float64:
                self.max_pixel_value = 1.0; self.img_type = np.float64
        # Initialize zero scene
        self.image = np.zeros(shape=(height, width), dtype=self.img_type)

    # %% Objects specification / generation
    @staticmethod
    def set_objects_props(mean_size: Union[float, int, tuple], n_objects: int = 2):
        pass

    # %% Put objects on the scene
    def put_objects_on(self, fluo_objects: tuple = ()):
        if len(fluo_objects) > 0:
            pass

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
    __acceptable_shape_types: list = ['round', 'r', 'ellipse', 'el', 'curved', 'c']  # shape types of the object
    __acceptable_border_types: list = ['precise', 'pr', 'computed', 'co']; radius: float = 0.0; a: float = 0.0; b: float = 0.0
    typical_sizes: tuple = ()  # for storing descriptive parameters for curve describing the shape of the object
    # below - storing names of implemented computing functions for the define continuous shape
    __acceptable_shape_methods = ['gaussian', 'g', 'lorentzian', 'lor', 'derivative of logistic func.', 'dlogf', 'bump square', 'bump2', 'bump cube',
                                  'bump3', 'bump ^8', 'bump8', 'smooth circle', 'smcir', 'oversampled circle', 'ovcir', 'undersampled circle', 'uncir']
    image_type = None; center_shifts: tuple = (0.0, 0.0)   # subpixel shift of the center of the object
    casted_profile: np.ndarray = None  # casted normalized profile to the provided image type
    __profile_croped: bool = False  # flag for setting if the profile was cropped (zero pixel rows / columns removed)
    __external_upper_coordinates: tuple = (0, 0)  # external image coordinates of the upper left pixel
    __external_image_sizes: tuple = (0, 0)   # sizes of an external image coordinates of the upper left pixel
    __within_image: bool = False  # flag for checking that the profile is still within the image

    def __init__(self, typical_size: Union[float, int, tuple], center_shifts: tuple = (0.0, 0.0), shape_type: str = 'round',
                 border_type: str = 'precise', shape_method: str = ''):
        f"""
        Initialize the class representation of a fluorescent object.

        The difference between used parameters can be observed by plotting of the calculated shapes (profiles) by get_shape() method.

        Parameters
        ----------
        typical_size : Union[float, int, tuple]
            Typical sizes of the object, e.g. for a bead - radius (float or int), for ellipse - tuple with axes a, b and angle in radians (3 values).
        center_shifts : tuple, optional
            Shifts in pixels of the object center, should be less than 1px. The default is (0.0, 0.0).
        shape_type : str, optional
            Supporeted shape types: {self.__acceptable_shape_types}. \n
            Currently implemented: 'round' or 'r' - for the circular bead object and 'ellipse' or 'el' - for the ellipse object.
            The default is 'round'.
        border_type : str, optional
            Type of intensity of the border pixels calculation. Supported border types: 'precise', 'pr', 'computed', 'co'. \n
            The 'computed' or 'co' type should be accomponied with the specification of the shape method parameter. \n
            The 'precise' or 'pr' type corresponds to the developed counting area of the pixel laying within the border (e.g., circular) of an object. \n
            Note that 'computed' type can be used only for 'round' objects. The default is 'precise'.
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
        # Checking typical sizes depending on the shape type
        if self.shape_type == "round" or self.shape_type == "r":
            typical_size = float(typical_size)  # assuming that the input parameter can be converted to the float type
            if typical_size < 0.5:
                raise ValueError(f"Expected typical size (radius) should be larger than 0.5px, provided: {typical_size}")
            else:
                self.radius = typical_size
        else:
            if self.shape_type == "ellipse" or self.shape_type == "el":
                self.border_type = 'precise'  # default border type for ellipse shape calculation
                if len(typical_size) != 3:
                    raise ValueError("For ellipse particle expected length of typical size tuple is equal 3: (a, b, angle)")
                else:
                    a, b, angle = typical_size; max_size = max(a, b); min_size = min(a, b)
                    if max_size < 1.0 or min_size < 0.5:
                        raise ValueError("Expected sizes a, b should be positive, minimal is more than 0.5px and maximal is more than 1px")
                    if angle > 2.0*np.pi or angle < -2.0*np.pi:
                        raise ValueError("Expected angle should be in the range of [-2pi, 2pi]")
                    self.typical_sizes = typical_size
            else:
                if len(typical_size) <= 2:
                    raise ValueError("For curved particle expected length of typical size tuple is more than 2")
                else:
                    self.typical_sizes = typical_size
            if self.shape_type == "el":
                self.explicit_shape_name = 'ellipse'
        # Assigning the explicit border type for making full naming for plots
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
                elif self.shape_method == 'ovcir':
                    self.explicit_shape_name = 'oversampled circle'
                elif self.shape_method == 'uncir':
                    self.explicit_shape_name = 'undersampled circle'
            else:
                raise ValueError(f"Provided shape computation method '{shape_method}' not in acceptable list {self.__acceptable_shape_methods}")
        # Check that explicit shape name is not empty
        if len(self.explicit_shape_name) == 0:
            if self.shape_type == "ellipse":
                self.explicit_shape_name = self.shape_type[:]
            elif self.border_type == "computed" or self.border_type == "co":
                self.explicit_shape_name = self.shape_method[:]
            elif self.border_type == "precise" or self.border_type == "pr":
                self.explicit_shape_name = "round"
        # Assuming that center shifts provided as the tuple with x shift, y shift floats in pixels
        if len(center_shifts) == 2:
            x_shift, y_shift = center_shifts
            if abs(x_shift) < 1.0 and abs(y_shift) < 1.0:
                self.center_shifts = center_shifts
            else:
                raise ValueError(f"One of the shifts '{center_shifts}' are more than 1px, but the shifts are expected to be in the subpixel range")

    # %% Calculate and plot shape
    def get_shape(self, center_shifts: tuple = None) -> np.ndarray:
        """
        Calculate and return 2D intensity normalized (to the range [0.0, 1.0]) distribution of the object shape.


        Parameters
        ----------
        center_shifts : tuple, optional
            Shifts in pixels of the object center, should be less than 1px. The default is None.

        Raises
        ------
        NotImplementedError
            For some set of allowed parameters for class initialization the calculation hasn't been yet implemented.

        Returns
        -------
        2D shape of the object (intensity representation).

        """
        if center_shifts is not None and len(center_shifts) == 2:
            x_shift, y_shift = center_shifts
            if abs(x_shift) < 1.0 and abs(y_shift) < 1.0:
                self.center_shifts = center_shifts
            else:
                raise ValueError(f"One of the shifts '{center_shifts}' are more than 1px, but the shifts are expected to be in the subpixel range")
        if (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "computed" or self.border_type == "co"):
            self.profile = continuous_shaped_bead(self.radius, self.center_shifts, bead_type=self.shape_method)
        elif (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "precise" or self.border_type == "pr"):
            self.profile = discrete_shaped_bead(self.radius, self.center_shifts)
        elif self.shape_type == "ellipse" or self.shape_type == "el":
            sizes = (self.typical_sizes[0], self.typical_sizes[1]); ellipse_angle = self.typical_sizes[2]
            self.profile = discrete_shaped_ellipse(sizes, ellipse_angle, self.center_shifts)
        else:
            raise NotImplementedError("This set of input parameters hasn't yet been implemented")

    def get_casted_shape(self, max_pixel_value: Union[int, float],
                         image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8') -> Union[np.ndarray, None]:
        """
        Calculate casted from the computed normalized object shape.

        Parameters
        ----------
        max_pixel_value : int | float (designated as Union)
            Maximum intensity or pixel value of the object.
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional
            Type for casting. The default is 'uint8'.

        Raises
        ------
        ValueError
            If the provided max_pixel_value doesn't correspond to the provided image type.

        Returns
        -------
        numpy.ndarray | None (designated as Union)
            Returns the casted profile or None if the normalized profile hasn't been calculated.

        """
        if image_type not in UscopeScene.acceptable_img_types:
            raise ValueError(f"Provided image type '{image_type}' not in the acceptable list of types: " + str(UscopeScene.acceptable_img_types))
        if self.profile is not None:
            if image_type == 'uint8' or image_type is np.uint8:
                if max_pixel_value > UscopeScene.max_pixel_value_uint8 or max_pixel_value < 0:
                    raise ValueError(f"Provided max pixel value {max_pixel_value} isn't compatible with the provided image type: {image_type}")
                self.image_type = image_type; self.casted_profile = np.round(max_pixel_value*self.profile, 0).astype(np.uint8)
                return self.casted_profile
            elif image_type == 'uint16' or image_type is np.uint16:
                if max_pixel_value > UscopeScene.max_pixel_value_uint16 or max_pixel_value < 0:
                    raise ValueError(f"Provided max pixel value {max_pixel_value} isn't compatible with the provided image type: {image_type}")
                self.image_type = image_type; self.casted_profile = np.round(max_pixel_value*self.profile, 0).astype(np.uint16)
                return self.casted_profile
            elif image_type == 'float' or image_type is np.float64:
                self.image_type = image_type; self.casted_profile = max_pixel_value*self.profile
                return self.casted_profile
        else:
            self.casted_profile = None; return self.casted_profile

    def crop_shape(self) -> np.ndarray:
        """
        Crop zero intensity parts (borders) of the computed profile.

        Returns
        -------
        numpy.ndarray
            2D cropped profile.

        """
        self.__profile_croped = False  # set it to the default value
        if self.profile is not None:
            m, n = self.profile.shape; i_start = 0; j_start = 0; i_end = m; j_end = n; border_found = False
            for i in range(m):
                if np.max(self.profile[i, :]) > 0.0:
                    if not border_found:
                        border_found = True
                    continue
                else:
                    if not border_found:
                        i_start = i
                    else:
                        i_end = i; break
            border_found = False
            for j in range(n):
                if np.max(self.profile[:, j]) > 0.0:
                    if not border_found:
                        border_found = True
                    continue
                else:
                    if not border_found:
                        j_start = j
                    else:
                        j_end = j; break
            self.profile = self.profile[i_start+1:i_end, j_start+1:j_end]
            if i_start > 0 or j_start > 0 or i_end < m or j_end < n:
                self.__profile_croped = True
        return self.profile

    # %% Plotting methods
    def plot_shape(self, str_id: str = ""):
        """
        Plot interactively the profile of the object computed by the get_shape() method along with the border of the object.

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
            if self.__profile_croped:
                naming = "Cropped Shape"
            else:
                naming = "Shape"
            plt.figure(f"{naming} with parameters: {self.explicit_shape_name}, {self.border_type}, center: {self.center_shifts} {str_id}")
            axes_img = plt.imshow(self.profile, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.colorbar(); plt.tight_layout()
            m_center, n_center = self.profile.shape
            if self.center_shifts[0] >= 0.0:
                m_center = m_center // 2 + self.center_shifts[0]
            else:
                if self.shape_type == "round" or self.shape_type == "r":
                    m_center = m_center // 2 + self.center_shifts[0] + 1.0
                else:
                    m_center = m_center // 2 + self.center_shifts[0]
            if self.center_shifts[1] >= 0.0:
                n_center = n_center // 2 + self.center_shifts[1]
            else:
                if self.shape_type == "round" or self.shape_type == "r":
                    n_center = n_center // 2 + self.center_shifts[1] + 1.0
                else:
                    n_center = n_center // 2 + self.center_shifts[1]
            if self.shape_type == "round" or self.shape_type == "r":
                axes_img.axes.add_patch(Circle((m_center, n_center), self.radius, edgecolor='red', linewidth=1.5, facecolor='none'))
                axes_img.axes.plot(m_center, n_center, marker='.', linewidth=3, color='red')
            elif self.shape_type == "ellipse" or self.shape_type == "el":
                axes_img.axes.add_patch(Ellipse((m_center, n_center), self.typical_sizes[0], self.typical_sizes[1],
                                                angle=-self.typical_sizes[2]*180.0/np.pi,
                                                edgecolor='red', facecolor='none', linewidth=1.75))
                axes_img.axes.plot(m_center, n_center, marker='.', linewidth=3, color='red')

    def plot_casted_shape(self, str_id: str = ""):
        """
        Plot interactively the casted to the provided type profile of the object computed by the get_casted_shape() method.

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
        if self.casted_profile is not None:
            plt.figure(f"Casted shape with parameters: {self.explicit_shape_name}, {self.border_type}, center: {self.center_shifts} {str_id}")
            plt.imshow(self.casted_profile, cmap='gray', origin='upper'); plt.axis('off'); plt.colorbar(); plt.tight_layout()

    # %% Containing checks
    def set_image_sizes(self, image_sizes: tuple):
        h, w = image_sizes  # expecting packed height, width - also can be called with image.shape (np.ndarray.shape) attribute
        if h > 2 and w > 2:
            self.__external_image_size = (h, w)
        else:
            raise ValueError("Height and width image is expected to be more than 2 pixels")


# %% Some tests
if __name__ == "__main__":
    plt.close("all"); test_computed_centered_beads = False; test_precise_centered_bead = False; test_computed_shifted_beads = False
    test_presice_shifted_beads = True; test_ellipse_centered = True; test_ellipse_shifted = True; test_casting = False
    test_put_objects = True; shifts = (-0.69, 0.44)

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
        gb9 = FluorObj(typical_size=2.0, border_type='co', shape_method='ovcir'); gb9.get_shape(); gb9.plot_shape()
        gb10 = FluorObj(typical_size=2.0, border_type='co', shape_method='uncir'); gb10.get_shape(); gb10.plot_shape()
    if test_precise_centered_bead:
        gb8 = FluorObj(typical_size=2.0); gb8.get_shape(); gb8.plot_shape()
    # Testing the shifted from the center objects generation
    if test_computed_shifted_beads and not test_computed_centered_beads:
        gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g', center_shifts=shifts); gb1.get_shape(); gb1.plot_shape()
        gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor', center_shifts=shifts); gb2.get_shape(); gb2.plot_shape()
        gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf', center_shifts=shifts); gb3.get_shape(); gb3.plot_shape()
        gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2', center_shifts=shifts); gb4.get_shape(); gb4.plot_shape()
        gb5 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump3', center_shifts=shifts); gb5.get_shape(); gb5.plot_shape()
        gb6 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump8', center_shifts=shifts); gb6.get_shape(); gb6.plot_shape()
        gb7 = FluorObj(typical_size=2.0, border_type='co', shape_method='smcir', center_shifts=shifts); gb7.get_shape(); gb7.plot_shape()
        gb9 = FluorObj(typical_size=2.0, border_type='co', shape_method='ovcir', center_shifts=shifts); gb9.get_shape(); gb9.plot_shape()
        gb10 = FluorObj(typical_size=2.0, border_type='co', shape_method='uncir', center_shifts=shifts); gb10.get_shape(); gb10.plot_shape()
    if test_presice_shifted_beads:
        gb8 = FluorObj(typical_size=2.0, center_shifts=shifts); gb8.get_shape(); gb8.crop_shape(); gb8.plot_shape()
        if test_casting:
            gb8.get_casted_shape(255, 'uint8'); gb8.plot_casted_shape()
    if test_ellipse_centered:
        gb11 = FluorObj(shape_type='ellipse', typical_size=(4.4, 2.5, np.pi/3)); gb11.get_shape(); gb11.plot_shape()
        if test_casting:
            gb11.get_casted_shape(255, 'uint8'); gb11.plot_casted_shape()
    if test_ellipse_shifted:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.crop_shape(); gb12.plot_shape()
        if test_casting:
            gb12.get_casted_shape(255, 'uint8'); gb12.plot_casted_shape()
    if test_put_objects:
        pass
