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
    max_pixel_value_uint8: int = 255; max_pixel_value_uint16: int = 65535; shape: tuple = (height, width)
    fluo_objects: list = []; shape_types = ['mixed', 'round', 'r', 'ellipse', 'el']
    __image_cleared: bool = True  # for tracking that the scene was cleared (zeroed)

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
        self.image = np.zeros(shape=(height, width), dtype=self.img_type); self.shape = (self.height, self.width); self.__image_cleared = True

    # %% Objects specification / generation
    @classmethod
    def get_random_objects(cls, mean_size: Union[float, int, tuple], size_std: Union[float, int, tuple], intensity_range: tuple, n_objects: int = 2,
                           shapes: str = 'round', image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8') -> tuple:
        """
        Generate objects with randomized shape sizes, for shapes: 'round', 'ellipse', 'mixed' - last one for randomized choice between 2 first ones.

        Objects are instances of FluorObj() class from this module.

        Parameters
        ----------
        mean_size : Union[float, int, tuple]
            Mean size(-s) of randomized objects. Integer or float is supposed to be used for round particles, tuple - for ellipse.
        size_std : Union[float, int, tuple]
            Standard deviation of mean size(-s).
        intensity_range : tuple
            (Min, Max) intensities for randomized choise of the maximum intensity along the profile.
        n_objects : int, optional
            Number of generated objects. The default is 2.
        shapes : str, optional
            Implemented in FluorObj() shapes: 'round', 'ellipse', 'mixed' of them. The default is 'round'.
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional
            Image type of the scene on which the objects will be placed and casted to. The default is 'uint8'.

        Raises
        ------
        ValueError
            See the provided description.

        Returns
        -------
        tuple
            Packed instances of FluorObj() class.

        """
        if shapes not in cls.shape_types:
            raise ValueError(f"Please provide the supported shape type for generation from a list: {cls.shape_types}")
        fl_objects = []  # for storing generated objects
        r = None; r_std = None  # default parameters for round objects
        min_intensity, max_intensity = intensity_range  # assuming that only 2 values provided
        for i in range(n_objects):
            if shapes == 'mixed':
                shape_type = random.choice(['round', 'ellipse'])
                # Define radius and radius std for round particles from the provided tuples for the ellipses (random choice between a and b)
                if isinstance(mean_size, tuple):
                    r = random.choice(mean_size)
                else:
                    r = mean_size
                if isinstance(size_std, tuple):
                    r_std = random.choice(size_std)
                else:
                    r_std = size_std
            else:
                shape_type = shapes
            # Random selection of central shifts for placement
            i_shift = round(random.random(), 3); j_shift = round(random.random(), 3)  # random generation of central pixel shifts
            # Checking that shifts are generated in the subpixel range and correcting it if not
            if i_shift >= 1.0:
                i_shift -= round(random.random(), 3)*0.25
            if j_shift >= 1.0:
                j_shift -= round(random.random(), 3)*0.25
            sign_i = random.choice([-1.0, 1.0]); sign_j = random.choice([-1.0, 1.0]); i_shift *= sign_i; j_shift *= sign_j  # random signs
            # Random selection of max intensity for the profile casting
            if isinstance(min_intensity, int) and isinstance(max_intensity, int):
                fl_intensity = random.randrange(min_intensity, max_intensity, 1)
            elif isinstance(min_intensity, float) and isinstance(max_intensity, float):
                fl_intensity = random.uniform(a=min_intensity, b=max_intensity)
            # Round shaped object generation
            if shape_type == 'round' or shape_type == 'r':
                if r is not None and r_std is not None:
                    radius = random.gauss(mu=r, sigma=r_std)
                else:
                    if isinstance(mean_size, tuple) or isinstance(size_std, tuple):
                        raise ValueError("Provided tuple with sizes for round shape object, there expected only single number size")
                    radius = random.gauss(mu=mean_size, sigma=size_std)
                # Checking generated radius for consistency
                if radius < 0.5:
                    radius += random.uniform(a=0.6-radius, b=0.6)
                fl_object = FluorObj(typical_size=radius, center_shifts=(i_shift, j_shift)); fl_object.get_shape(); fl_object.crop_shape()
                fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type); fl_objects.append(fl_object)
            # Ellipse shaped object generation
            elif shape_type == 'ellipse' or shape_type == 'el':
                a, b = mean_size; a_std, b_std = size_std  # unpacking tuples assuming 2 of sizes packed there
                a_r = random.gauss(mu=a, sigma=a_std); b_r = random.gauss(mu=b, sigma=b_std)
                angle = random.uniform(a=0.0, b=2.0*np.pi)  # get random orientation for an ellipse
                # Checking generated a, b axes for consistency (min axis >= 0.5, max axis >= 1.0)
                if a < 0.5:
                    a += random.uniform(0.6-a, 0.6)
                elif b < 0.5:
                    b += random.uniform(0.6-b, 0.6)
                max_axis = max(a, b)
                if max_axis < 1.0:
                    if a == max_axis:
                        a += random.uniform(1.1-a, 1.1)
                    else:
                        b += random.uniform(1.1-b, 1.1)
                fl_object = FluorObj(typical_size=(a_r, b_r, angle), center_shifts=(i_shift, j_shift), shape_type='ellipse')
                fl_object.get_shape(); fl_object.crop_shape()  # calculate normalized shape and crop it
                fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type); fl_objects.append(fl_object)
        return tuple(fl_objects)

    def set_random_places(self, fluo_objects: tuple = (), not_overlapping: bool = True) -> tuple:
        # TODO: implement not_overlapping generation logic
        if len(fluo_objects) > 0:
            h, w = self.image.shape  # height and width of the image for setting the range of coordinates random selection
            for fluo_obj in fluo_objects:
                if fluo_obj.profile is not None:
                    h_fl_obj, w_fl_obj = fluo_obj.profile.shape  # get object sizes
                    fluo_obj.set_image_sizes(self.image.shape)  # force to account for the used scene shape
                    i_smallest = -(h_fl_obj // 2); j_smallest = -(w_fl_obj // 2)  # smallest placing coordinate = half of object sizes
                    i_largest = h - h_fl_obj // 2 - 1; j_largest = w - w_fl_obj // 2 - 1
                    i = random.randrange(i_smallest, i_largest, 1); j = random.randrange(j_smallest, j_largest, 1)
                    fluo_obj.set_coordinates((i, j))  # set random place of the object within the image
        return fluo_objects

    # %% Put objects on the scene
    def put_objects_on(self, fluo_objects: tuple = (), save_objects: bool = True, rewrite_objects: bool = False):
        """
        Put the provided objects on the scene by checking pixelwise the provided profiles and in the case of intersection of
        two objects profiles, selecting the maximum intensity from them.

        Parameters
        ----------
        fluo_objects : tuple, optional
            Fluorescent objects, instances of FluorObj class, packed in a tuple. The default is ().
        save_objects : bool, optional
            If True, will save (append) objects in the class attribute 'fluo_objects'. The default is True. \n
            Note that, if it's False, then before placing the objects, the scene will be cleared (stored before objects will be removed from it).
        rewrite_objects : bool, optional
            If True, it forces to substitute stored objects in the class attribute 'fluo_objects' with the provided ones. The default is False.

        Returns
        -------
        None.

        """
        if len(fluo_objects) > 0:
            if len(self.fluo_objects) == 0 or not save_objects:
                self.clear_scene()  # clear the scene before placing the objects only if there are no saved objects and objects provided not for saving
            for fluo_obj in fluo_objects:
                if fluo_obj.within_image and fluo_obj.casted_profile is not None:
                    i_start, j_start = fluo_obj.external_upper_coordinates; i_size, j_size = fluo_obj.casted_profile.shape
                    h, w = self.image.shape  # for checking that the placing happening within the image
                    k = 0; m = 0  # for counting on the profile
                    for i in range(i_start, i_start+i_size):
                        m = 0  # refresh starting column for counting on the object profile
                        for j in range(j_start, j_start+j_size):
                            if 0 <= i < h and 0 <= j < w:
                                if self.img_type == np.uint8 or self.img_type == np.uint16:
                                    if self.image[i, j] == 0:
                                        self.image[i, j] = fluo_obj.casted_profile[k, m]
                                    else:
                                        self.image[i, j] = max(self.image[i, j], fluo_obj.casted_profile[k, m])
                                else:  # float image type
                                    if self.image[i, j] < 1E-9:
                                        self.image[i, j] = fluo_obj.casted_profile[k, m]
                                    else:
                                        self.image[i, j] = max(self.image[i, j], fluo_obj.casted_profile[k, m])
                                if self.__image_cleared:
                                    self.__image_cleared = False
                            m += 1  # next column of the casted profile
                        k += 1  # next row of the casted profile
            if save_objects:
                l_fluo_objects = list(fluo_objects)  # conversion input tuple to the list
                if not rewrite_objects:
                    self.fluo_objects = self.fluo_objects + l_fluo_objects  # concatenate 2 lists
                else:
                    self.fluo_objects = l_fluo_objects[:]  # copy the list

    def spread_objects_on(self, fluo_objects: tuple):
        """
        Compose 2 subsequent methods: self.put_objects_on(slef.set_random_places(fluo_objects), save_objects=True).

        So, this method puts the provided objects, which are randomly spread on the scene, and saves them in the attribute.

        Parameters
        ----------
        fluo_objects : tuple
            Fluorescent objects, instances of FluorObj class, packed in a tuple.

        Returns
        -------
        None.

        """
        self.put_objects_on(fluo_objects=self.set_random_places(fluo_objects))

    def recreate_scene(self):
        if len(self.fluo_objects) > 0:
            pass

    # %% Scene manipulation
    def show_scene(self, color_map='viridis', str_id: str = ""):
        """
        Show interactively the stored in the class scene (image) by plotting it using matplotlib.

        Parameters
        ----------
        color_map
            Color map acceptable by matplotlib.pyplot.cm. Fallback is viridis color map. The default is 'viridis'.
        str_id : str, optional
            Unique string id for plotting several plots with unique Figure() names. The default is "".

        Returns
        -------
        None.

        """
        h, w = self.image.shape; height_width_ratio = h/w; default_image_size = 5.8
        if len(str_id) == 0:
            str_id = str(random.randint(1, 100))
        if not plt.isinteractive():
            plt.ion()
        if self.__image_cleared:
            figure_name = "Blank UscopeScene " + str_id
        else:
            figure_name = "UscopeScene " + str_id
        plt.figure(figure_name, figsize=(default_image_size, default_image_size*height_width_ratio))
        try:
            plt.imshow(self.image, cmap=color_map, origin='upper')
        except ValueError:
            plt.imshow(self.image, cmap=plt.cm.viridis, origin='upper')
        plt.axis('off'); plt.tight_layout()

    def clear_scene(self):
        """
        Reinitialize the scene to zero (plain dark background).

        Returns
        -------
        None.

        """
        self.image = np.zeros(shape=(self.height, self.width), dtype=self.img_type); self.__image_cleared = True

    def is_blank(self) -> bool:
        """
        Returns True, if the scene (image) is blank (zeroed).

        Returns
        -------
        bool
            Value for designation of a blank scene.

        """
        return self.__image_cleared


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
    external_upper_coordinates: tuple = (0, 0)  # external image coordinates of the upper left pixel
    __external_image_sizes: tuple = (0, 0)   # sizes of an external image coordinates of the upper left pixel
    within_image: bool = False  # flag for checking that the profile is still within the image

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
            if self.casted_profile is not None:
                self.casted_profile = self.casted_profile[i_start+1:i_end, j_start+1:j_end]
            if i_start > 0 or j_start > 0 or i_end < m or j_end < n:
                self.__profile_croped = True
        return self.profile

    # %% Plotting methods
    def plot_shape(self, str_id: str = ""):
        """
        Plot interactively the profile of the object computed by the get_shape() method along with the border of the object.

        Please note that the border will be plotted if the the shape hadn't been cropped or in some cases if it had.

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
            plot_patch = True  # flag for plotting the patch (Circle or Ellipse)
            if not self.__profile_croped:
                m_center, n_center = self.profile.shape  # image sizes
                m_center = m_center // 2 + self.center_shifts[0]; n_center = n_center // 2 + self.center_shifts[1]
                if self.center_shifts[0] < 0.0 and (self.shape_type == "round" or self.shape_type == "r"):
                    m_center = m_center // 2 + self.center_shifts[0] + 1.0
                if self.center_shifts[1] < 0.0 and (self.shape_type == "round" or self.shape_type == "r"):
                    n_center = n_center // 2 + self.center_shifts[1] + 1.0
            else:
                if self.shape_type == "ellipse" or self.shape_type == "el":
                    plot_patch = False
                m_center, n_center = self.profile.shape; m_center = m_center // 2; n_center = n_center // 2
                if 0.0 < self.center_shifts[0] <= 0.5 or -0.5 <= self.center_shifts[0] < 0.0:
                    m_center += self.center_shifts[0]
                else:
                    plot_patch = False  # prevent plotting patch because after cropping the shape can be shifted on the plot
                if 0.0 < self.center_shifts[1] <= 0.5 or -0.5 <= self.center_shifts[1] < 0.0:
                    n_center += self.center_shifts[1]
                else:
                    plot_patch = False
            if plot_patch:
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

    # %% Containing within the image checks
    def set_image_sizes(self, image_sizes: tuple):
        """
        Set containing this object image sizes packed in the tuple.

        Parameters
        ----------
        image_sizes : tuple
            Expected as (height, width).

        Raises
        ------
        ValueError
            If image sizes provided less than 2 pixels.

        Returns
        -------
        None.

        """
        h, w = image_sizes  # expecting packed height, width - also can be called with image.shape (np.ndarray.shape) attribute
        if h > 2 and w > 2:
            self.__external_image_sizes = (h, w)
        else:
            self.__external_image_sizes = (0, 0)  # put the default sizes for always getting False after setting coordinates
            raise ValueError("Height and width image is expected to be more than 2 pixels")

    def set_coordinates(self, coordinates: tuple) -> bool:
        """
        Set coordinates of the left upper coordinates of the profile.

        Parameters
        ----------
        coordinates : tuple
            As i - from rows, j - columns.

        Returns
        -------
        bool
            True if the image containing the profile (even partially).

        """
        i, j = coordinates; self.external_upper_coordinates = coordinates  # expecting packed coordinates of the left upper pixel of the profile
        if self.profile is not None:
            h, w = self.profile.shape; h_image, w_image = self.__external_image_sizes
            i_border_check = False; j_border_check = False  # flags for overall checking
            if (i < 0 and i + h >= 0) or (i >= 0 and i < h_image):
                i_border_check = True
            if (j < 0 and j + w >= 0) or (j >= 0 and j < w_image):
                j_border_check = True
            self.within_image = (i_border_check and j_border_check)
        else:
            self.within_image = False  # explicit setting flag
        return self.within_image


# %% Some tests
if __name__ == "__main__":
    plt.close("all"); test_computed_centered_beads = False; test_precise_centered_bead = False; test_computed_shifted_beads = False
    test_presice_shifted_beads = False; test_ellipse_centered = False; test_ellipse_shifted = False; test_casting = False
    test_cropped_shapes = False; test_put_objects = False; test_generate_objects = False; test_overall_gen = True; shifts = (-0.2, 0.44)

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
    if test_cropped_shapes:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.plot_shape()
    if test_put_objects:
        scene = UscopeScene(width=24, height=21); gb8 = FluorObj(typical_size=2.78, center_shifts=shifts); gb8.get_shape()
        gb12 = FluorObj(shape_type='el', typical_size=(4.9, 2.8, np.pi/3)); gb12.get_shape(); gb12.get_casted_shape(max_pixel_value=255)
        gb8.get_casted_shape(max_pixel_value=254); gb8.crop_shape(); gb12.crop_shape()  # gb8.plot_shape(); gb12.plot_shape()
        gb8.set_image_sizes(scene.shape); gb12.set_image_sizes(scene.shape); gb8.set_coordinates((-1, 3)); gb12.set_coordinates((12, 5))
        gb9 = FluorObj(typical_size=3.0, center_shifts=(0.25, -0.1)); gb9.get_shape(); gb9.get_casted_shape(max_pixel_value=251)
        gb9.crop_shape(); gb9.set_image_sizes(scene.shape); gb9.set_coordinates((10, 18))
        scene.put_objects_on(fluo_objects=(gb8, gb12)); scene.put_objects_on(fluo_objects=(gb9, )); scene.show_scene()
    if test_generate_objects:
        objs = UscopeScene.get_random_objects(mean_size=4.2, size_std=1.5, shapes='r', intensity_range=(230, 252), n_objects=2)
        objs2 = UscopeScene.get_random_objects(mean_size=(7.3, 5), size_std=(2, 1.19), shapes='el', intensity_range=(220, 250), n_objects=2)
        scene = UscopeScene(width=45, height=38); objs = scene.set_random_places(objs); objs2 = scene.set_random_places(objs2)
        scene.put_objects_on(objs); scene.put_objects_on(objs2, save_objects=False); scene.show_scene()
        # scene2 = UscopeScene(width=55, height=46); scene2.show_scene()  # will show 'Blank' image
    if test_overall_gen:
        objs3 = UscopeScene.get_random_objects(mean_size=(8.3, 5.4), size_std=(2, 1.19), shapes='mixed', intensity_range=(182, 250), n_objects=5)
        scene = UscopeScene(width=55, height=46); scene.spread_objects_on(objs3); scene.show_scene(color_map='gray')
