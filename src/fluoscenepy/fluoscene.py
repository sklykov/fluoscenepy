# -*- coding: utf-8 -*-
"""
Main script for the 'fluoscenepy' package.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Global imports
import random
import time
import warnings
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

try:
    import numba
    if numba is not None:
        numba_installed = True
    from numba import njit
except ModuleNotFoundError:
    numba_installed = False

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead, discrete_shaped_ellipse
    from utils.comp_funcs import (get_random_shape_props, get_random_central_shifts, get_random_max_intensity, get_radius_gaussian,
                                  get_ellipse_sizes, print_out_elapsed_t, delete_coordinates_from_list, set_binary_mask_coords_in_loop)
    if numba_installed:
        from utils.compiled_objects_gen import discrete_shaped_bead_acc, discrete_shaped_ellipse_acc
        from utils.acc_comp_funcs import generate_coordinates_list, set_binary_mask_coordinates
else:
    from .utils.raw_objects_gen import continuous_shaped_bead, discrete_shaped_bead, discrete_shaped_ellipse
    from .utils.comp_funcs import (get_random_shape_props, get_random_central_shifts, get_random_max_intensity, get_radius_gaussian,
                                   get_ellipse_sizes, print_out_elapsed_t, delete_coordinates_from_list, set_binary_mask_coords_in_loop)
    if numba_installed:
        from .utils.compiled_objects_gen import discrete_shaped_bead_acc, discrete_shaped_ellipse_acc
        from .utils.acc_comp_funcs import generate_coordinates_list, set_binary_mask_coordinates


# %% Scene (image) class def.
class UscopeScene:
    """
    Replicate the common fluorescence microscopic image (frame or 'scene'). \n

    This class simulates the bright objects with round and elongate shapes as the basic examples of imaged objects. \n
    The ratio behind development of this class - to get the ground truth images for image processing workflows tests. \n
    For more complex and useful cases, please, check the References.

    References
    ----------
    In general, there is no references for implementation. For the used noise models, see the 'add_noise method'.

    """

    # Class parameters
    width: int = 4; height: int = 4; __warn_message: str = ""; max_pixel_value = 255; img_type = np.uint8
    acceptable_img_types: list = ['uint8', 'uint16', 'float', np.uint8, np.uint16, np.float64]
    max_pixel_value_uint8: int = 255; max_pixel_value_uint16: int = 65535; shape: tuple = (height, width)
    fluo_objects: list = []; shape_types = ['mixed', 'round', 'r', 'ellipse', 'el']; image: np.ndarray = None
    __image_cleared: bool = True  # for tracking that the scene was cleared (zeroed)
    __available_coordinates: list = []; __binary_placement_mask: np.ndarray = None; denoised_image: np.ndarray = None
    __restricted_coordinates: list = []; __noise_added: bool = False  # for tracking that the noise has been added

    def __init__(self, width: int, height: int, image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8',
                 numba_precompile: bool = True):
        """
        Initialize the class wrapper for store the current "scene" or microscopic image.

        Parameters
        ----------
        width : int \n
            Width of the scene in pixels. \n
        height : int \n
            Height of the scene in pixels. \n
        image_type : str | np.uint8 | np.uint16 | np.float64, optional \n
            Image type, for supported ones see the acceptable types. The default is 'uint8'. \n
        numba_precompile : bool, optional \n
            Flag for precompilation of computing methods using 'numba' library (if it's installed). The default is True.

        Raises
        ------
        ValueError \n
            If provided parameters are inconsistent, e.g. width or height < 2. \n

        Returns
        -------
        None.

        """
        # Check provided width and height
        if width < 4 or height < 4:
            raise ValueError(f"Provided dimensions ({width}x{height}) is less than 4")
        if width > 10000 or height > 10000:
            __warn_message = f"Provided dimensions ({width}x{height}) pixels are unrealistic for the common image"
            warnings.warn(__warn_message)
        self.width = width; self.height = height
        # Check an image type
        if image_type not in self.acceptable_img_types:
            raise ValueError(f"Provided image type '{image_type}' not in the acceptable list of types: " + str(self.acceptable_img_types))
        else:
            if image_type == 'uint8' or image_type == np.uint8:
                self.max_pixel_value = self.max_pixel_value_uint8; self.img_type = np.uint8
            elif image_type == 'uint16' or image_type == np.uint16:
                self.max_pixel_value = self.max_pixel_value_uint16; self.img_type = np.uint16
            elif image_type == 'float' or image_type == np.float64:
                self.max_pixel_value = 1.0; self.img_type = np.float64
        # Initialize zero scene
        self.image = np.zeros(shape=(height, width), dtype=self.img_type)
        self.__image_cleared = True; self.shape = (self.height, self.width)
        self.methods_precompiled = False  # flag for retaining if the objects shape calculation methods precompiled
        # Precompiling of calculation methods for making calculations more convenient
        if numba_precompile and numba_installed:
            self.precompile_methods()

    # %% Objects specification / generation
    @classmethod
    def get_random_objects(cls, mean_size: Union[float, int, tuple], size_std: Union[float, int, tuple], intensity_range: tuple,
                           n_objects: int = 2, shapes: str = 'round', image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8',
                           verbose_info: bool = False, accelerated: bool = False) -> tuple:
        """
        Generate objects with randomized shape sizes, for shape types: 'round', 'ellipse', 'mixed' - last one for randomized choice
        between 2 first ones.

        Objects should be instances of FluorObj() class from this module.

        Parameters
        ----------
        mean_size : Union[float, int, tuple] \n
            Mean size(-s) of randomized objects. Integer or float is supposed to be used for round particles, tuple - for ellipse. \n
        size_std : Union[float, int, tuple] \n
            Standard deviation of mean size(-s). \n
        intensity_range : tuple \n
            (Min, Max) intensities for randomized choice of the maximum intensity along the profile. \n
        n_objects : int, optional \n
            Number of generated objects. The default is 2. \n
        shapes : str, optional \n
            Implemented in FluorObj() shapes: 'round', 'ellipse' or 'mixed' them (randomly selected). The default is 'round'. \n
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional \n
            Image type of the scene on which the objects will be placed and cast to. The default is 'uint8'. \n
        verbose_info : bool, optional \n
            Flag for printing out verbose information about the generation progress (use it for many objects generation).
            The default is False. \n
        accelerated : bool, optional \n
            Flag for attempting acceleration of objects shape calculation by using 'numba' library compilation. \n

        Raises
        ------
        ValueError  \n
            See the provided description in the exception traceback. \n

        Returns
        -------
        tuple
            Packed instances of FluorObj() class with the generated objects.

        """
        if verbose_info:
            t_ov_1 = time.perf_counter()
        # Check input parameters for consistency
        if shapes not in cls.shape_types:
            raise ValueError(f"Please provide the supported shape type for generation from a list: {cls.shape_types}")
        max_intensity = max(intensity_range); raise_exception = False  # define max provided intensity
        if image_type == 'uint8' or image_type == np.uint8:
            if max_intensity > cls.max_pixel_value_uint8:
                raise_exception = True
        elif image_type == 'uint16' or image_type == np.uint16:
            if max_intensity > cls.max_pixel_value_uint16:
                raise_exception = True
        if raise_exception:
            raise ValueError(f"Max intensity from the range {intensity_range} is incompatible with the provided image type {image_type}")
        fl_objects = []  # for storing generated objects
        r = None; r_std = None  # default parameters for round objects
        # Printout warning if the method called with parameters leading to the long calculations
        if verbose_info and not accelerated:
            if np.min(mean_size) >= 4.0 and n_objects >= 10:
                print(f"***** Note that the overall generation will take more than ~ {round(1.075*n_objects, 1)} sec. *****", flush=True)
        elif verbose_info and accelerated:
            if np.min(mean_size) >= 4.0 and n_objects >= 34:
                print(f"***** Note that the overall generation will take more than ~ {round(0.3*n_objects, 1)} sec. *****", flush=True)
        # Generation loop
        for i in range(n_objects):
            if verbose_info:
                t1 = time.perf_counter()
                print(f"Started generation of #{i+1} object", flush=True)
            if shapes == 'mixed':
                shape_type, r, r_std = get_random_shape_props(mean_size, size_std)
            else:
                shape_type = shapes
            # Random selection of object parameters: shifts, max intensity
            i_shift, j_shift = get_random_central_shifts(); fl_intensity = get_random_max_intensity(intensity_range)
            # Round shaped object generation
            if shape_type == 'round' or shape_type == 'r':
                radius = get_radius_gaussian(r, r_std, mean_size, size_std)  # Gaussian-distributed random value
                # Generating the object and calculating its shape, cast and crop it
                fl_object = FluorObj(typical_size=2.0*radius, center_shifts=(i_shift, j_shift)); fl_object.get_shape(accelerated=accelerated)
                fl_object.crop_shape(); fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type)
                fl_objects.append(fl_object)
            # Ellipse shaped object generation
            elif shape_type == 'ellipse' or shape_type == 'el':
                ellipse_sizes = get_ellipse_sizes(mean_size, size_std)
                fl_object = FluorObj(typical_size=ellipse_sizes, center_shifts=(i_shift, j_shift), shape_type='ellipse')
                fl_object.get_shape(accelerated=accelerated); fl_object.crop_shape()  # calculate normalized shape and crop it
                fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type); fl_objects.append(fl_object)
            if verbose_info:
                print_out_elapsed_t(t1, operation=f"Generation of obj. #{i+1} out of {n_objects}")
        if verbose_info:
            print_out_elapsed_t(t_ov_1, operation="Overall generation")
        return tuple(fl_objects)

    def get_objects_acc(self, mean_size: Union[float, int, tuple], size_std: Union[float, int, tuple], intensity_range: tuple,
                        n_objects: int = 2, shapes: str = 'round', image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8',
                        verbose_info: bool = False) -> tuple:
        """
        Generate objects with randomized shape sizes, for shape types: 'round', 'ellipse', 'mixed' - last one for randomized choice
        between 2 first ones.

        This method is accelerated by utilizing the 'numba' library compilation of called inside methods. \n
        Objects should be instances of FluorObj() class from this module.

        Parameters
        ----------
        mean_size : Union[float, int, tuple]  \n
            Mean size(-s) of randomized objects. Integer or float is supposed to be used for round particles, tuple - for ellipse. \n
        size_std : Union[float, int, tuple]  \n
            Standard deviation of mean size(-s). \n
        intensity_range : tuple  \n
            (Min, Max) intensities for randomized choice of the maximum intensity along the profile.  \n
        n_objects : int, optional  \n
            Number of generated objects. The default is 2.  \n
        shapes : str, optional  \n
            Implemented in FluorObj() shapes: 'round', 'ellipse' or 'mixed' them (randomly selected). The default is 'round'.   \n
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional  \n
            Image type of the scene on which the objects will be placed and cast to. The default is 'uint8'.  \n
        verbose_info : bool, optional  \n
            Flag for printing out verbose information about the generation progress (use it for many objects generation).
            The default is False. \n

        Raises
        ------
        ValueError \n
            See the provided description in the exception traceback. \n

        Returns
        -------
        tuple
            Packed instances of FluorObj() class with the generated objects.

        """
        if verbose_info:
            t_ov_1 = time.perf_counter()
        # Checking input values for consistency
        if shapes not in self.shape_types:
            raise ValueError(f"Please provide the supported shape type for generation from a list: {cls.shape_types}")
        if not numba_installed:
            raise ValueError(f"Method can be called if only 'numba' package is installed")
        max_intensity = max(intensity_range); raise_exception = False  # define max provided intensity
        if image_type == 'uint8' or image_type == np.uint8:
            if max_intensity > self.max_pixel_value_uint8:
                raise_exception = True
        elif image_type == 'uint16' or image_type == np.uint16:
            if max_intensity > self.max_pixel_value_uint16:
                raise_exception = True
        if raise_exception:
            raise ValueError(f"Max intensity from the range {intensity_range} is incompatible with the provided image type {image_type}")
        # Generation parameters
        fl_objects = []  # for storing generated objects
        r = None; r_std = None  # default parameters for round objects
        # Generation loop
        for i in range(n_objects):
            if verbose_info:
                t1 = time.perf_counter()
                print(f"Started generation of #{i+1} object", flush=True)
            if shapes == 'mixed':
                shape_type, r, r_std = get_random_shape_props(mean_size, size_std)
            else:
                shape_type = shapes
            # Random selection of object parameters: shifts, max intensity
            i_shift, j_shift = get_random_central_shifts(); fl_intensity = get_random_max_intensity(intensity_range)
            # Round shaped object generation
            if shape_type == 'round' or shape_type == 'r':
                radius = get_radius_gaussian(r, r_std, mean_size, size_std)  # Gaussian-distributed random value
                # Generating the object and calculating its shape, cast and crop it
                fl_object = FluorObj(typical_size=2.0*radius, center_shifts=(i_shift, j_shift)); fl_object.get_shape(accelerated=True)
                fl_object.crop_shape(); fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type)
                fl_objects.append(fl_object)
            # Ellipse shaped object generation
            elif shape_type == 'ellipse' or shape_type == 'el':
                ellipse_sizes = get_ellipse_sizes(mean_size, size_std)
                fl_object = FluorObj(typical_size=ellipse_sizes, center_shifts=(i_shift, j_shift), shape_type='ellipse')
                fl_object.get_shape(accelerated=True); fl_object.crop_shape()  # calculate normalized shape and crop it
                fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type); fl_objects.append(fl_object)
            if verbose_info:
                print_out_elapsed_t(t1, operation=f"Generation of obj. #{i+1} out of {n_objects}")
        if verbose_info:
            print_out_elapsed_t(t_ov_1, operation="Overall generation")
        return tuple(fl_objects)

    def precompile_methods(self, verbose_info: bool = False) -> bool:
        """
        Precompile computation methods in 'get_random_objects' chain of methods calls.

        Parameters
        ----------
        verbose_info : bool, optional  \n
            Flag for more prints out about calculations timing and status. The default is False. \n

        Returns
        -------
        bool  \n
            - True, if methods have been precompiled (prerequisite - 'numba' library installed).

        """
        if not self.methods_precompiled:
            if verbose_info:
                t_ov_1 = time.perf_counter()
                # print("*****Precompilation of shape generation methods started*****")
            self.__round_fl_obj = FluorObj(typical_size=2.02, center_shifts=(0.1, -0.2)); self.__round_fl_obj.get_shape(accelerated=True)
            self.__ellipse_fl_obj = FluorObj(typical_size=(2.02, 1.52, np.pi/3.0), center_shifts=(-0.1, 0.2), shape_type='ellipse')
            self.__ellipse_fl_obj.get_shape(accelerated=True); self.methods_precompiled = True
            if verbose_info:
                print_out_elapsed_t(t_ov_1, "*** Precompilation")
        return self.methods_precompiled

    @staticmethod
    def get_round_objects(mean_size: float, size_std: float, intensity_range: tuple, n_objects: int = 2, shape_r_type: str = 'mixed',
                          image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8') -> tuple:
        """
        Generate round shaped objects (as FluorObj class instances) with continuous shapes (see FluorObj class documentation).

        Parameters
        ----------
        mean_size : float  \n
            Mean size of objects in pixels.  \n
        size_std : float  \n
            Standard deviation of objects sizes. \n
        intensity_range : tuple \n
            Intensity range for selection of maximum object intensity. \n
        n_objects : int, optional \n
            Number of generated objects. The default is 2. \n
        shape_r_type : str, optional \n
            Acceptable by FluorObj class round and computed shape types. The default is 'mixed'. \n
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional \n
            Type of image used for a scene. The default is 'uint8'. \n

        Returns
        -------
        tuple  \n
            Generated objects.

        """
        fl_objects = []; min_intensity, max_intensity = intensity_range
        shape_types = FluorObj.valuable_round_shapes[:]; random.shuffle(shape_types)
        for i in range(n_objects):
            # Random selection of central shifts for placement
            i_shift = round(random.random(), 3); j_shift = round(random.random(), 3)  # random generation of central pixel shifts
            # Checking that shifts are generated in the subpixel range and correcting it if not
            if i_shift >= 1.0:
                i_shift -= round(random.random(), 3)*0.25
            if j_shift >= 1.0:
                j_shift -= round(random.random(), 3)*0.25
            radius = abs(random.gauss(mu=mean_size, sigma=size_std))  # get the abs. value of radius from Gaussian distribution
            # Random selection of max intensity for the profile casting
            if isinstance(min_intensity, int) and isinstance(max_intensity, int):
                fl_intensity = random.randrange(min_intensity, max_intensity, 1)
            elif isinstance(min_intensity, float) and isinstance(max_intensity, float):
                fl_intensity = random.uniform(a=min_intensity, b=max_intensity)
            # Checking generated radius for consistency
            if radius < 0.5:
                radius += random.uniform(a=0.51-radius, b=.51)
            # If mixed type, selecting randomly the shape type
            if shape_r_type == 'mixed':
                shape_sel_type = random.choice(shape_types)
            else:
                shape_sel_type = shape_r_type
            # Generating the round shaped object with continuous function used for shape calculation
            fl_object = FluorObj(typical_size=2.0*radius, center_shifts=(i_shift, j_shift), border_type='co', shape_method=shape_sel_type)
            fl_object.get_shape(); fl_object.crop_shape()  # calculate normalized shape and crop it
            fl_object.get_casted_shape(max_pixel_value=fl_intensity, image_type=image_type); fl_objects.append(fl_object)
        return tuple(fl_objects)

    # %% Randomly assigning coordinates for objects
    def set_random_places(self, fluo_objects: tuple = (), overlapping: bool = True, touching: bool = True,
                          only_within_scene: bool = False, verbose_info: bool = False) -> tuple:
        """
        Set random coordinates within scene (or partially outside) for provided objects.

        Parameters
        ----------
        fluo_objects : tuple, optional \n
            Tuple with generated instances of FluorObj() class. The default is (). \n
        overlapping : bool, optional \n
            If True, the coordinates will be randomly selected from available ones. \n
            If False, the largest object will be placed firstly, and others will be placed after with checking that the object to be placed
            are not intersected with the already placed objects on previous iterations.  \n
            The default is True. \n
        touching : bool, optional \n
            Flag for allowing objects to touch by their border pixels. Note that if overlapping is True, this flag ignored.
            The default is True. \n
        only_within_scene : bool, optional
            If True, the objects will be placed completely within (inside) the scene. \n
            If False, objects may lay partially outside the scene. The default is False. \n
        verbose_info : bool, optional \n
            Printing out verbose information about performance of placement. The default is False. \n

        Returns
        -------
        tuple
            With the placed objects.

        """
        if len(fluo_objects) > 0:
            # For making verbose information printouts, check the time performance of placement logic
            if verbose_info:
                t1 = time.perf_counter(); n_objects = len(fluo_objects)
            h, w = self.image.shape  # height and width of the image for setting the range of coordinates random selection
            if not overlapping:
                filtered_fluo_obj = []  # storing the objects in the list for excluding the not placed objects if they are not overlapped
                additional_border = 0
                if not touching:
                    additional_border = 1  # additional 1 pixel for excluding the pixels close to the object borders
            # If only_within_scene, sort out the objects based on their sizes and place first the largest one, if not - the smallest one
            if not overlapping or (overlapping and only_within_scene):
                fluo_objects = list(fluo_objects)  # convert for applying embedded sorting algorithm
            if only_within_scene:
                fluo_objects.sort(reverse=True)
            elif not overlapping and not only_within_scene:
                fluo_objects.sort()
            # Printing out the warning messages (obtained for some useful cases and flags)
            if len(fluo_objects) > 1 and verbose_info and fluo_objects[0].profile is not None:
                h_fl_obj, w_fl_obj = fluo_objects[0].profile.shape  # get the largest object sizes
                if len(fluo_objects) >= 5 or (h_fl_obj > 10 and w_fl_obj > 10):
                    if not numba_installed:
                        print("NOTE: Placing algorithm can take quite a long time (depending on the size of objects and scene). \n"
                              + "For acceleration of it install library 'numba' with the recommended version >= 0.57.1", flush=True)
                    print(f"***** Placing of {len(fluo_objects)} objects started *****", flush=True)
            if verbose_info and not overlapping:
                if not numba_installed:
                    if h*w > 35000:
                        print("Note that placing algorithm can take long time, starting from several dozens of seconds", flush=True)
                else:
                    if h*w > 110000:
                        print("Note that placing algorithm can take long time, starting from several dozens of seconds", flush=True)
            # Placing loop for provided objects (set random placing coordinates of the upper left pixel for the objects)
            placed_objects = 0  # for printing out the number of placed objects
            for fluo_obj in fluo_objects:
                if fluo_obj.profile is not None:
                    if verbose_info:
                        t_obj = time.perf_counter(); print(f"Started placing of #{placed_objects+1} object", flush=True)
                    h_fl_obj, w_fl_obj = fluo_obj.profile.shape  # get object sizes
                    fluo_obj.set_image_sizes(self.image.shape)  # force to account for the used scene shape
                    # Placement logic, if overlapping is allowed - just random spreading objects on the scene
                    if overlapping or (not overlapping and self.__binary_placement_mask is None):
                        # Depending on the flag, defining smallest and largest coordinates (outside the scene borders or only inside)
                        if only_within_scene:
                            i_smallest = 1; j_smallest = 1; i_largest = h-h_fl_obj-1; j_largest = w-w_fl_obj-1
                        else:
                            # smallest placing coordinate = half of object sizes - 1 pixel, analogous for largest coordinates
                            i_smallest = -(h_fl_obj // 2) + 1; j_smallest = -(w_fl_obj // 2) + 1
                            i_largest = (h - h_fl_obj // 2) - 2; j_largest = (w - w_fl_obj // 2) - 2
                        # Random selection of i, j coordinates below
                        i_obj = random.randrange(i_smallest, i_largest, 1); j_obj = random.randrange(j_smallest, j_largest, 1)
                        # Not overlapping, below - place the largest object and create the mask for preventing the overlapping
                        if not overlapping:
                            # Generate the meshgrid of all available for placing coordinates (depending on the placement flag)
                            if not only_within_scene and len(fluo_objects) > 1:
                                if not numba_installed:
                                    self.__available_coordinates = [(i_a, j_a) for i_a in range(i_smallest, i_largest)
                                                                    for j_a in range(j_smallest, j_largest)]
                                else:
                                    precomp_l = generate_coordinates_list(1, 3, 1, 3)  # precompile method for acceleration further calls
                                    self.__available_coordinates = generate_coordinates_list(i_smallest, i_largest, j_smallest, j_largest)
                            else:
                                # Below - keep the possible placement close to bottom and right edges of the scene and remove them later
                                if not numba_installed:
                                    self.__available_coordinates = [(i_a, j_a) for i_a in range(i_smallest, h-2)
                                                                    for j_a in range(j_smallest, w-2)]
                                    self.__restricted_coordinates = [(i_a, j_a) for i_a in range(i_smallest, i_largest)
                                                                     for j_a in range(j_smallest, j_largest)]
                                else:
                                    precomp_l = generate_coordinates_list(1, 3, 1, 3)  # precompile method for acceleration further calls
                                    self.__available_coordinates = generate_coordinates_list(i_smallest, h-2, j_smallest, w-2)
                                    self.__restricted_coordinates = generate_coordinates_list(i_smallest, i_largest, j_smallest, j_largest)
                            # Generate the binary mask (instead of the real image or scene) for placing more than 1 object
                            if len(fluo_objects) > 1:
                                self.__binary_placement_mask = np.zeros(shape=self.image.shape, dtype='uint8')
                                if not numba_installed:
                                    self.__binary_placement_mask = set_binary_mask_coords_in_loop(self.__binary_placement_mask,
                                                                                                  i_obj-additional_border,
                                                                                                  i_obj+h_fl_obj+additional_border,
                                                                                                  self.image.shape[0],
                                                                                                  j_obj-additional_border,
                                                                                                  j_obj+w_fl_obj+additional_border,
                                                                                                  self.image.shape[1])
                                else:
                                    precompl_mask = set_binary_mask_coordinates(np.zeros((2, 2)), 0, 1, 2, 0, 1, 2)  # precompilation
                                    self.__binary_placement_mask = set_binary_mask_coordinates(self.__binary_placement_mask,
                                                                                               i_obj-additional_border,
                                                                                               i_obj+h_fl_obj+additional_border,
                                                                                               self.image.shape[0],
                                                                                               j_obj-additional_border,
                                                                                               j_obj+w_fl_obj+additional_border,
                                                                                               self.image.shape[1])
                                # Exclude the coordinates occupied by the placed object from the available choices for further placements
                                if not numba_installed:
                                    coordinates_for_del = [(i_exc, j_exc)
                                                           for i_exc in range(i_obj-additional_border, i_obj+h_fl_obj+additional_border)
                                                           for j_exc in range(j_obj-additional_border, j_obj+w_fl_obj+additional_border)]
                                else:
                                    coordinates_for_del = generate_coordinates_list(i_obj-additional_border, i_obj+h_fl_obj+additional_border,
                                                                                    j_obj-additional_border, j_obj+w_fl_obj+additional_border)
                                self.__available_coordinates = delete_coordinates_from_list(coordinates_for_del,
                                                                                            self.__available_coordinates)
                            filtered_fluo_obj.append(fluo_obj)  # storing the 1st placed and not overlapped objects
                            random.shuffle(self.__available_coordinates)  # shuffle available coordinates
                        fluo_obj.set_coordinates((i_obj, j_obj)); placed_objects += 1  # set random place of the object within the image
                        if verbose_info:
                            print("# of placed objects:", placed_objects, "out of", len(fluo_objects), flush=True)
                            print_out_elapsed_t(t_obj, operation="Placing 1st object")
                    else:
                        # Trying to place the object in the randomly selected from remaining coordinates place and checking if there is no
                        # intersections with already placed objects, regulate # of attempts to place below in the while condition
                        if verbose_info:
                            t_obj = time.perf_counter()
                        i_attempts = 0; placed = False; available_correcting_coordinates = self.__available_coordinates[:]
                        # Adapting max number of attempts
                        if 0 < len(available_correcting_coordinates) < 50:
                            max_attempts = len(available_correcting_coordinates)
                        else:
                            max_attempts = 50  # limiting max number of attempts to find suitable coordinates
                        # Trying to place the object and check for intersections with the other ones
                        border_coordinates_deleted = False
                        while (not placed and i_attempts < max_attempts) and len(available_correcting_coordinates) > 0:
                            overlapped = False  # flag for checking if the object is overlapped with the occupied place on a binary mask
                            # randomly choose from the available coordinates
                            if not only_within_scene:
                                i_obj, j_obj = random.choice(available_correcting_coordinates)
                            else:
                                if not border_coordinates_deleted:
                                    i_obj, j_obj = random.choice(self.__restricted_coordinates)
                                else:
                                    i_obj, j_obj = random.choice(available_correcting_coordinates)
                            # Vectorized form of the overlapping check
                            i_max, j_max = self.__binary_placement_mask.shape; i_max -= 1; j_max -= 1
                            # Define the region of the placement mask for checking
                            if i_obj-additional_border < 0:
                                i_start = 0
                            else:
                                i_start = i_obj - additional_border
                            if j_obj-additional_border < 0:
                                j_start = 0
                            else:
                                j_start = j_obj - additional_border
                            if i_obj + h_fl_obj + additional_border > i_max:
                                i_finish = i_max
                            else:
                                i_finish = i_obj + h_fl_obj + additional_border
                            if j_obj + w_fl_obj + additional_border > j_max:
                                j_finish = j_max
                            else:
                                j_finish = j_obj + w_fl_obj + additional_border
                            if np.max(self.__binary_placement_mask[i_start:i_finish, j_start:j_finish]) > 0:
                                overlapped = True; placed = False; i_attempts += 1
                                try:
                                    available_correcting_coordinates.remove((i_obj, j_obj))
                                    self.__available_coordinates.remove((i_obj, j_obj))
                                except ValueError:
                                    pass
                            # Check that object placed and if not, delete border pixels for placing (long-lasting function)
                            if not overlapped:
                                placed = True; break
                            else:
                                # Prevent placing out of scene by removing from available coordinates
                                if only_within_scene:
                                    if not numba_installed:
                                        coordinates_for_del = [(i_exc, j_exc) for i_exc in range(h-h_fl_obj, h-2) for j_exc in range(1, w-2)]
                                        coordinates_for_del += [(i_exc, j_exc) for i_exc in range(1, h-2) for j_exc in range(w-w_fl_obj, w-2)]
                                    else:
                                        coordinates_for_del = generate_coordinates_list(h-h_fl_obj, h-2, 1, w-2)
                                        coordinates_for_del += generate_coordinates_list(1, h-2, w-w_fl_obj, w-2)
                                    available_correcting_coordinates = delete_coordinates_from_list(coordinates_for_del,
                                                                                                    available_correcting_coordinates)
                                    border_coordinates_deleted = True
                            if len(available_correcting_coordinates) == 0:
                                placed = False; break
                        # Exclude the coordinates occupied by the placed object from the meshgrid (list with coordinates pares)
                        if placed:
                            if not numba_installed:
                                coordinates_for_del = [(i_exc, j_exc)
                                                       for i_exc in range(i_obj-additional_border, i_obj+h_fl_obj+additional_border)
                                                       for j_exc in range(j_obj-additional_border, j_obj+w_fl_obj+additional_border)]
                            else:
                                coordinates_for_del = generate_coordinates_list(i_obj-additional_border, i_obj+h_fl_obj+additional_border,
                                                                                j_obj-additional_border, j_obj+w_fl_obj+additional_border)
                            for del_coord in coordinates_for_del:
                                try:
                                    self.__available_coordinates.remove(del_coord)
                                    if only_within_scene:
                                        self.__restricted_coordinates.remove(del_coord)
                                except ValueError:
                                    pass
                            # Exclude placed object from binary placement mask
                            if not numba_installed:
                                self.__binary_placement_mask = set_binary_mask_coords_in_loop(self.__binary_placement_mask,
                                                                                              i_obj, i_obj+h_fl_obj,
                                                                                              self.image.shape[0],
                                                                                              j_obj, j_obj+w_fl_obj,
                                                                                              self.image.shape[1])
                            else:
                                self.__binary_placement_mask = set_binary_mask_coordinates(self.__binary_placement_mask,
                                                                                           i_obj, i_obj+h_fl_obj,
                                                                                           self.image.shape[0],
                                                                                           j_obj, j_obj+w_fl_obj,
                                                                                           self.image.shape[1])
                            fluo_obj.set_coordinates((i_obj, j_obj))  # if found place, place the object
                            filtered_fluo_obj.append(fluo_obj); placed_objects += 1  # collect for returning only placed objects
                            if verbose_info:
                                print("# of placed objects:", placed_objects, "out of", len(fluo_objects), flush=True)
                                print_out_elapsed_t(t_obj, operation=f"Placing #{placed_objects} object")
            if not overlapping:
                fluo_objects = tuple(filtered_fluo_obj)  # convert list -> tuple for returning only placed objects
                # plt.figure("Binary Placement Mask"); plt.imshow(self.__binary_placement_mask)  # plot the occupied places by the objects
            if verbose_info:
                print_out_elapsed_t(t1, operation=f"Placing of {placed_objects} objects")
        return fluo_objects

    # %% Put objects on the scene
    def put_objects_on(self, fluo_objects: tuple = (), save_objects: bool = True, save_only_objects_inside: bool = False,
                       rewrite_objects: bool = False):
        """
        Put the provided objects on the scene by checking pixelwise the provided profiles and in the case of intersection of
        two objects profiles, selecting the maximum intensity from them.

        Parameters
        ----------
        fluo_objects : tuple, optional  \n
            Fluorescent objects, instances of FluorObj class, packed in a tuple. The default is (). \n
        save_objects : bool, optional \n
            If True, will save (append) objects in the class attribute 'fluo_objects'. The default is True. \n
            Note that, if it's False, then before placing the objects, the scene will be cleared
            (stored before objects will be removed from it). \n
        save_only_objects_inside : bool, optional \n
            Save in the class attribute ('fluo_objects') only objects that are inside the image. The default is False. \n
        rewrite_objects : bool, optional \n
            If True, it forces to substitute stored objects in the class attribute 'fluo_objects' with the provided ones.
            The default is False. \n

        Returns
        -------
        None.

        """
        if len(fluo_objects) > 0:
            if len(self.fluo_objects) == 0 or not save_objects:
                self.clear_scene()  # clear the scene before placing the objects only if there are no saved objects and the flag is False
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
                            m += 1  # next column of the cast profile
                        k += 1  # next row of the cast profile
            if save_objects:
                l_fluo_objects = list(fluo_objects)  # conversion input tuple to the list
                if save_only_objects_inside:
                    l_fluo_objects = [fluo_obj for fluo_obj in l_fluo_objects if fluo_obj.within_image]
                if not rewrite_objects:
                    self.fluo_objects = self.fluo_objects + l_fluo_objects  # concatenate 2 lists
                else:
                    self.fluo_objects = l_fluo_objects[:]  # copy the list

    def spread_objects_on(self, fluo_objects: tuple):
        """
        Compose 2 subsequent methods: self.put_objects_on(self.set_random_places(fluo_objects), save_objects=True).

        So, this method puts the provided objects, which are randomly spread on the scene, and saves them in the attribute.

        Parameters
        ----------
        fluo_objects : tuple \n
            Fluorescent objects, instances of FluorObj class, packed in a tuple. \n

        Returns
        -------
        None.

        """
        self.put_objects_on(fluo_objects=self.set_random_places(fluo_objects))

    def recreate_scene(self):
        """
        Recreate the saved objects in the class attribute 'fluo_objects' on the cleared before scene.

        Returns
        -------
        None.

        """
        if len(self.fluo_objects) > 0 and self.__image_cleared:
            self.put_objects_on(fluo_objects=tuple(self.fluo_objects), save_objects=False)
        elif not self.__image_cleared:
            if len(self.__warn_message) == 0:
                self.__warn_message = "The scene is not clear, cannot recreate the scene"
                warnings.warn(self.__warn_message)
            elif self.__warn_message == "The scene is not clear, cannot recreate the scene":
                self.__warn_message = ""
        elif len(self.fluo_objects) == 0:
            if len(self.__warn_message) == 0:
                self.__warn_message = "There are no stored objects within this class instance"
                warnings.warn(self.__warn_message)
            elif self.__warn_message == "There are no stored objects within this class instance":
                self.__warn_message = ""

    # %% Scene manipulation
    def show_scene(self, str_id: str = "", color_map='viridis', unique_plot_id: bool = True):
        """
        Show interactively the stored in the class scene (image) by plotting it using matplotlib.

        Parameters
        ----------
        str_id : str, optional \n
            Unique string id for plotting several plots with unique Figure() names. The default is "". \n
        color_map \n
            Color map acceptable by matplotlib.pyplot.cm. Fallback is viridis color map. The default is 'viridis'. \n
        unique_plot_id: bool, optional \n
            Flag for adding to plot name some random integer id for preventing plots overlapping. The default is True. \n

        Returns
        -------
        None.

        """
        h, w = self.image.shape; height_width_ratio = h/w; default_image_size = 5.8; additional_id = ""
        if len(str_id) == 0:
            str_id = str(random.randint(1, 1000))
        elif unique_plot_id:
            additional_id = str(random.randint(1, 1000))
        if not plt.isinteractive():
            plt.ion()
        if self.__image_cleared:
            figure_name = "Blank UscopeScene " + str_id + " " + additional_id
        else:
            figure_name = "UscopeScene " + str_id + " " + additional_id
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
        if not self.__image_cleared:
            self.image = np.zeros(shape=(self.height, self.width), dtype=self.img_type)
            self.__binary_placement_mask = None; self.__image_cleared = True

    def is_blank(self) -> bool:
        """
        Returns True, if the scene (image) is blank (zeroed).

        Returns
        -------
        bool \n
            Value for designation of a blank scene. \n

        """
        return self.__image_cleared

    # %% Making scene realistic and useful by adding noise: shot and detector ones
    def add_noise(self, seed: int = None, mean_noise: Union[int, float] = None, sigma_noise: Union[int, float] = None) -> np.ndarray:
        """
        Add Poisson (signal dependent shot noise) and background (Gaussian or detector) noise.

        Note that if the called again on the scene, this method will add newly generated noise to the initial image.

        Parameters
        ----------
        seed : int, optional \n
            Long integer for initializing pseudorandom generator for repeating generated sequences. The default is None. \n
        mean_noise : Union[int, float], optional \n
            Mean for Gaussian noise intensity. The default is None. \n
        sigma_noise : Union[int, float], optional \n
            Sigma for Gaussian noise intensity. The default is None. \n

        References
        ----------
        [1] "Imaging in focus: An introduction to denoising bioimages in the era of deep learning", R.F. Laine,
        G. Jacquemet, A. Krull (2021) \n
        [2] Online Resource: https://bioimagebook.github.io/chapters/3-fluorescence/3-formation_noise/formation_noise.html  \n

        Returns
        -------
        numpy.ndarray \n
            The scene added shot (Poisson) and detector (Gaussian) noises. \n

        """
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        if not self.__image_cleared:
            # Add the Gaussian noise (background)
            if mean_noise is None:
                mean_noise = 0.125*np.max(self.image)  # 12.5% of the max value
            if sigma_noise is None:
                sigma_noise = 0.33*mean_noise  # 33% of the mean
            noisy_background = rng.normal(mean_noise, sigma_noise, size=self.image.shape)  # normally distributed noise on background
            noisy_background = np.where(noisy_background < 0.0, 0.0, noisy_background)  # check that all pixel values are positive
            # Substitute the calculated exact pixel value with the Poisson distributed one due to the properties of commonly used cameras
            h, w = self.image.shape; raw_pixels = np.zeros(shape=self.image.shape)
            if not self.__noise_added:
                self.denoised_image = self.image.copy()  # copy initial scene without noise
            else:
                self.image = self.denoised_image.copy()  # restore the initial image by the copying the content of the denoised image
            # Generate and add noise pixelwise
            for i in range(h):
                for j in range(w):
                    if float(self.image[i, j]) > 0.0:
                        raw_pixels[i, j] = rng.poisson(lam=self.image[i, j])  # save converted original value with the Poisson-distributed
                        raw_pixels[i, j] += noisy_background[i, j]  # add Gaussian noise
                        if raw_pixels[i, j] > self.max_pixel_value:
                            raw_pixels[i, j] = self.max_pixel_value  # check that the pixel values are still in range with the image type
                    else:
                        raw_pixels[i, j] = noisy_background[i, j]
            # Casting the noisy image back to the original image format
            if self.img_type == 'uint16' or self.img_type is np.uint16 or self.img_type == 'uint8' or self.img_type is np.uint8:
                self.image = np.round(raw_pixels, 0).astype(self.img_type)
            else:
                self.image = raw_pixels.astype(self.img_type)
            self.__noise_added = True  # switch the flag for tracking that the noise has been added
        return self.image

    def remove_noise(self) -> np.ndarray:
        """
        Remove added previously noise by copying the stored in 'denoise_image' pixel content.

        Returns
        -------
        numpy.ndarray \n
            Initial scene without noise. \n

        """
        if self.__noise_added and self.denoised_image is not None:
            self.image = self.denoised_image.copy(); self.__noise_added = False; self.denoised_image = None
        return self.image

    @staticmethod
    def noise2image(image: np.ndarray, seed: int = None, mean_noise: Union[int, float] = None,
                    sigma_noise: Union[int, float] = None) -> np.ndarray:
        source_image = np.copy(image)  # to guarantee that input image is not modified
        # Initialize random generator using provided seed
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        # Add the Gaussian noise (background)
        if mean_noise is None:
            mean_noise = 0.125*np.max(source_image)  # 12.5% of the max value
        if sigma_noise is None:
            sigma_noise = 0.33*mean_noise  # 33% of the mean
        noisy_background = rng.normal(mean_noise, sigma_noise, size=source_image.shape)  # normally distributed noise on background
        noisy_background = np.where(noisy_background < 0.0, 0.0, noisy_background)  # check that all pixel values are positive
        # Substitute the calculated exact pixel value with the Poisson distributed one due to the properties of commonly used cameras
        h, w = source_image.shape; raw_pixels = np.zeros(shape=source_image.shape, dtype=np.float64)
        # Define max pixel value
        if source_image.dtype == np.uint8:
            max_pixel_value = 255
        elif source_image.dtype == np.uint16:
            max_pixel_value = 2**16 - 1
        elif source_image.dtype == np.float64:
            max_pixel_value = 1.0
        else:
            raise ValueError("Supported image types: uint8, uint16 and float64. \n Convert the provided image to these types.")
        # Generate and add noise pixelwise, checking the calculated pixel value for consistency with the image type
        for i in range(h):
            for j in range(w):
                if float(source_image[i, j]) > 0.0:
                    raw_pixels[i, j] = rng.poisson(lam=source_image[i, j])  # save converted original value with the Poisson-distributed
                    raw_pixels[i, j] += noisy_background[i, j]  # add Gaussian noise
                    if raw_pixels[i, j] > max_pixel_value:
                        raw_pixels[i, j] = max_pixel_value  # check that the pixel values are still in range with the image type
                else:
                    raw_pixels[i, j] = noisy_background[i, j]
        # Casting the noisy image back to the original image format
        if source_image.dtype == np.uint8 or source_image.dtype == np.uint16:
            noisy_image = np.round(raw_pixels, 0).astype(source_image.dtype)
        else:
            noisy_image = raw_pixels.astype(source_image.dtype)
        return noisy_image


# %% Object class definition
class FluorObj:
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
    __acceptable_shape_methods: list = ['gaussian', 'g', 'lorentzian', 'lor', 'derivative of logistic func.', 'dlogf', 'bump square',
                                        'bump2', 'bump cube', 'bump3', 'bump ^8', 'bump8', 'smooth circle', 'smcir', 'oversampled circle',
                                        'ovcir', 'undersampled circle', 'uncir', 'circle', 'c']
    valuable_round_shapes: list = ['gaussian', 'derivative of logistic func.', 'bump cube', 'bump ^8', 'smooth circle']
    image_type = None; center_shifts: tuple = (0.0, 0.0)   # subpixel shift of the center of the object
    casted_profile: np.ndarray = None  # cast normalized profile to the provided image type
    __profile_cropped: bool = False  # flag for setting if the profile was cropped (zero pixel rows / columns removed)
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
        typical_size : Union[float, int, tuple] \n
            Typical sizes of the object, e.g. for a bead - diameter (float or int), for ellipse - tuple with axes a, b
            and angle in radians (3 values). \n
        center_shifts : tuple, optional \n
            Shifts in pixels of the object center, should be less than 1px. The default is (0.0, 0.0). \n
        shape_type : str, optional \n
            Supported shape types: {self.__acceptable_shape_types}. \n
            Currently implemented: 'round' or 'r' - for the circular bead object and 'ellipse' or 'el' - for the ellipse object.
            The default is 'round'. \n
        border_type : str, optional \n
            Type of intensity of the border pixels calculation. Supported border types: 'precise', 'pr', 'computed', 'co'. \n
            The 'computed' or 'co' type should be accompanied with the specification of the shape method parameter. \n
            The 'precise' or 'pr' type corresponds to the developed counting area of the pixel laying within
            the border (e.g., circular) of an object. \n
            Note that 'computed' type can be used only for 'round' objects. The default is 'precise'. \n
        shape_method : str, optional \n
            Shape method calculation, supported ones: {self.__acceptable_shape_methods}. The default is ''. \n

        Raises
        ------
        ValueError \n
            See the provided error description for details. \n

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
            if typical_size < 1.0:
                raise ValueError(f"Expected typical size (radius) should be larger than 1px, provided: {typical_size}")
            else:
                self.radius = 0.5*typical_size
        else:
            if self.shape_type == "ellipse" or self.shape_type == "el":
                self.border_type = 'precise'  # default border type for ellipse shape calculation
                if len(typical_size) != 3:
                    raise ValueError("For ellipse particle expected length of typical size tuple is equal 3: (a, b, angle)")
                else:
                    a, b, angle = typical_size; max_size = max(a, b); min_size = min(a, b)
                    if max_size < 1.5 or min_size < 1.0:
                        raise ValueError("Expected sizes a, b should be positive, minimal is more than 1px and maximal is more than 1.5px")
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
                elif self.shape_method == 'c':
                    self.explicit_shape_name = 'circle'
            else:
                raise ValueError(f"Provided shape comp. method '{shape_method}' not in acceptable list {self.__acceptable_shape_methods}")
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
                raise ValueError(f"One of the shifts '{center_shifts}' are more than 1px, but they are expected to be in the subpixel range")

    def get_shaping_functions(self) -> str:
        """
        Print out the implemented methods acceptable for the 'shape_method' parameter.

        Returns
        -------
        str \n
            Printed out composed informational string. \n

        """
        methods = ""; i = 0
        while i <= len(self.__acceptable_shape_methods)-1:
            methods += "'" + self.__acceptable_shape_methods[i] + "'" + " or " + "'" + self.__acceptable_shape_methods[i+1] + "'" + "\n"
            i += 2
        info_str = ("The implemented types of continuous bell shape functions or acceptable variables for the 'shape_method' parameter "
                    + "(first - the long naming followed by the shortened version): \n" + methods)
        print(info_str, flush=True)
        return info_str

    # %% Calculate and plot shape
    def get_shape(self, center_shifts: tuple = None, accelerated: bool = False) -> np.ndarray:
        """
        Calculate and return 2D intensity normalized (to the range [0.0, 1.0]) distribution of the object shape.


        Parameters
        ----------
        center_shifts : tuple, optional \n
            Shifts in pixels of the object center, should be less than 1px. The default is None. \n
        accelerated : bool, optional \n
            Accelerate the computation by using numba library compilation utilities.
            If True, will raise warning if the numba library hasn't been installed in the environment. The default is False. \n

        Raises
        ------
        NotImplementedError \n
            For some set of allowed parameters for class initialization the calculation hasn't been yet implemented. \n

        References
        ----------
        Continuously shaped objects are created by adapting the functions from: \n
        [1] https://en.wikipedia.org/wiki/Bell-shaped_function \n
        'Precisely' shaped objects are created by the custom algorithm, which calculates part of an object still laying
        within the affected ('border') pixels. \n

        Returns
        -------
        2D shape of the object (intensity representation).

        """
        if center_shifts is not None and len(center_shifts) == 2:
            x_shift, y_shift = center_shifts
            if abs(x_shift) < 1.0 and abs(y_shift) < 1.0:
                self.center_shifts = center_shifts
            else:
                raise ValueError(f"One of the shifts '{center_shifts}' are more than 1px, "
                                 + "but the shifts are expected to be in the subpixel range")
        if (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "computed" or self.border_type == "co"):
            self.profile = continuous_shaped_bead(self.radius, self.center_shifts, bead_type=self.shape_method)
        elif (self.shape_type == "round" or self.shape_type == "r") and (self.border_type == "precise" or self.border_type == "pr"):
            if not accelerated:
                self.profile = discrete_shaped_bead(self.radius, self.center_shifts)
            elif numba_installed:
                self.profile = discrete_shaped_bead_acc(self.radius, self.center_shifts)
            else:
                if accelerated and not numba_installed:
                    __warn_message = "Acceleration isn't possible because 'numba' library not installed in the current environment"
                    warnings.warn(__warn_message)
                self.profile = discrete_shaped_bead(self.radius, self.center_shifts)
        elif self.shape_type == "ellipse" or self.shape_type == "el":
            sizes = (self.typical_sizes[0], self.typical_sizes[1]); ellipse_angle = self.typical_sizes[2]
            if not accelerated:
                self.profile = discrete_shaped_ellipse(sizes, ellipse_angle, self.center_shifts)
            elif numba_installed:
                self.profile = discrete_shaped_ellipse_acc(sizes, ellipse_angle, self.center_shifts)
            else:
                if accelerated and not numba_installed:
                    __warn_message = "Acceleration isn't possible because 'numba' library not installed in the current environment"
                    warnings.warn(__warn_message)
                self.profile = discrete_shaped_ellipse(sizes, ellipse_angle, self.center_shifts)
        else:
            raise NotImplementedError("This set of input parameters hasn't yet been implemented")
        self.__profile_cropped = False  # set it to the default value, the profile is uncropped

    def get_casted_shape(self, max_pixel_value: Union[int, float],
                         image_type: Union[str, np.uint8, np.uint16, np.float64] = 'uint8') -> Union[np.ndarray, None]:
        """
        Calculate cast from the computed normalized object shape.

        Parameters
        ----------
        max_pixel_value : int | float (designated as Union) \n
            Maximum intensity or pixel value of the object. \n
        image_type : Union[str, np.uint8, np.uint16, np.float64], optional \n
            Type for casting. The default is 'uint8'. \n

        Raises
        ------
        ValueError \n
            If the provided max_pixel_value doesn't correspond to the provided image type. \n

        Returns
        -------
        numpy.ndarray | None (designated as Union) \n
            Returns the cast profile or None if the normalized profile hasn't been calculated. \n

        """
        if image_type not in UscopeScene.acceptable_img_types:
            raise ValueError(f"Provided image type '{image_type}' not in the acceptable list of types: "
                             + str(UscopeScene.acceptable_img_types))
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
        numpy.ndarray \n
            2D cropped profile. \n

        """
        if self.profile is not None and not self.__profile_cropped:  # checking that the profile has been calculated and not cropped yet
            m, n = self.profile.shape; i_start = 0; j_start = 0; i_end = m; j_end = n; border_found = False
            for i in range(m):
                if np.max(self.profile[i, :]) > 0.0:
                    if not border_found:
                        border_found = True
                    continue
                else:
                    if not border_found:
                        i_start = i+1
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
                        j_start = j+1
                    else:
                        j_end = j; break
            self.profile = self.profile[i_start:i_end, j_start:j_end]
            if self.casted_profile is not None:
                self.casted_profile = self.casted_profile[i_start:i_end, j_start:j_end]
            if i_start > 0 or j_start > 0 or i_end < m or j_end < n:
                self.__profile_cropped = True
        return self.profile

    # %% Plotting methods
    def plot_shape(self, str_id: str = "", color_map='viridis'):
        """
        Plot interactively the profile of the object computed by the get_shape() method along with the border of the object.

        Please note that the border will be plotted if the shape hadn't been cropped before.

        Parameters
        ----------
        str_id : str, optional \n
            Unique string id for plotting several plots with unique Figure() names. The default is "". \n
        color_map \n
            Color map acceptable by matplotlib.pyplot.cm. Fallback is viridis color map. The default is 'viridis'. \n

        Returns
        -------
        None.

        """
        if not plt.isinteractive():
            plt.ion()
        if self.profile is not None:
            if self.__profile_cropped:
                naming = "Cropped Shape"
            else:
                naming = "Shape"
            if len(str_id) == 0:
                str_id = str(random.randint(1, 1000))
            plt.figure(f"{naming} with parameters: {self.explicit_shape_name}, {self.border_type}, center: {self.center_shifts} {str_id}")
            try:
                axes_img = plt.imshow(self.profile, cmap=color_map, origin='upper')
            except ValueError:
                axes_img = plt.imshow(self.profile, cmap=plt.cm.viridis, origin='upper')
            plt.axis('off'); plt.colorbar(); plt.tight_layout()
            plot_patch = True  # flag for plotting the patch (Circle or Ellipse)
            if not self.__profile_cropped:
                m_center, n_center = self.profile.shape  # image sizes
                if self.center_shifts[0] < 0.0 and (self.shape_type == "round" or self.shape_type == "r"):
                    m_center = m_center // 2 + self.center_shifts[0] + 1.0
                elif self.center_shifts[0] >= 0.0:
                    m_center = m_center // 2 + self.center_shifts[0]
                if self.center_shifts[1] < 0.0 and (self.shape_type == "round" or self.shape_type == "r"):
                    n_center = n_center // 2 + self.center_shifts[1] + 1.0
                elif self.center_shifts[1] >= 0.0:
                    n_center = n_center // 2 + self.center_shifts[1]
            else:
                plot_patch = False  # just prevent the plotting the patch on the cropped images
            if plot_patch:
                if self.shape_type == "round" or self.shape_type == "r":
                    axes_img.axes.add_patch(Circle((m_center, n_center), self.radius, edgecolor='red', linewidth=1.5, facecolor='none'))
                    axes_img.axes.plot(m_center, n_center, marker='.', linewidth=3, color='red')
                elif self.shape_type == "ellipse" or self.shape_type == "el":
                    axes_img.axes.add_patch(Ellipse((m_center, n_center), self.typical_sizes[0], self.typical_sizes[1],
                                                    angle=-self.typical_sizes[2]*180.0/np.pi,
                                                    edgecolor='red', facecolor='none', linewidth=1.75))
                    axes_img.axes.plot(m_center, n_center, marker='.', linewidth=3, color='red')

    def plot_cast_shape(self, str_id: str = ""):
        """
        Plot interactively the cast to the provided type profile of the object computed by the get_casted_shape() method.

        Parameters
        ----------
        str_id : str, optional \n
            Unique string id for plotting several plots with unique Figure() names. The default is "". \n

        Returns
        -------
        None.

        """
        if not plt.isinteractive():
            plt.ion()
        if self.casted_profile is not None:
            if len(str_id) == 0:
                str_id = str(random.randint(1, 1000))
            plt.figure(f"Casted shape with parameters: {self.explicit_shape_name}, {self.border_type}, center: {self.center_shifts} {str_id}")
            plt.imshow(self.casted_profile, cmap='gray', origin='upper'); plt.axis('off'); plt.colorbar(); plt.tight_layout()

    # %% Containing within the image checks
    def set_image_sizes(self, image_sizes: tuple):
        """
        Set containing this object image sizes packed in the tuple.

        Parameters
        ----------
        image_sizes : tuple \n
            Expected as (height, width). \n

        Raises
        ------
        ValueError \n
            If image sizes provided less than 2 pixels. \n

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
        coordinates : tuple \n
            As i - from rows, j - columns. \n

        Returns
        -------
        bool \n
            True if the image containing the profile (even partially). \n

        """
        # Expecting packed coordinates of the left upper pixel coordinates of a profile
        i, j = coordinates; self.external_upper_coordinates = coordinates
        if self.profile is not None:
            h, w = self.profile.shape; h_image, w_image = self.__external_image_sizes
            i_border_check = False; j_border_check = False  # flags for overall checking
            # Below - basic check that the profile still is within image, if even its borders still are in
            if (i < 0 <= i + h) or (i >= 0 and i - h < h_image):
                i_border_check = True
            if (j < 0 <= j + w) or (j >= 0 and j - w < w_image):
                j_border_check = True
            # Additional check that pixels of a shape is still not zero within the image (touching the scene only by border pixels)
            if (i < 0 and j < 0) or (i > h_image and j > w_image):
                additional_containing_check = False
                for i_c in range(i, i+h):
                    k = 0
                    for j_c in range(j, j+w):
                        m = 0
                        if 0 <= i_c < h_image and 0 <= j_c < w_image:
                            if self.profile[k, m] > 0.0:
                                additional_containing_check = True; break
                        m += 1
                    if additional_containing_check:
                        break
                    k += 1
            else:
                additional_containing_check = True
            self.within_image = (i_border_check and j_border_check and additional_containing_check)
        else:
            self.within_image = False  # explicit setting flag
        return self.within_image

    # %% Rewriting dunder methods for implementing sorting logic
    def __lt__(self, other) -> bool:
        """
        Implementation of '<' comparison operator for letting the default sorting method to work on list of instances.

        This method compares only shape sizes multiplication (m, n = self.profile.shape; m*n < other.profile.shape[0]*other.profile.shape[1]).

        Parameters
        ----------
        other : FluorObj \n
            Instance of FluorObj() comparison. \n

        Returns
        -------
        bool \n
            Result of profile sizes comparison. \n

        """
        if self.profile is not None and other.profile is not None:
            return self.profile.shape[0]*self.profile.shape[1] < other.profile.shape[0]*other.profile.shape[1]
        else:
            return False

    def __eq__(self, other) -> bool:
        """
        Implementation of '==' comparison operator for letting the default sorting method to work on list of instances.

        This method compares shape size (m, n = self.profile.shape; m == other.profile.shape[0] and n == other.profile.shape[1])
        and also explicit_shape_name attributes of two classes.

        Parameters
        ----------
        other : FluorObj  \n
            Instance of FluorObj() comparison. \n

        Returns
        -------
        bool \n
            Result of mentioned above attributes comparison. \n

        """
        if self.profile is not None and other.profile is not None:
            shape_type_check = self.explicit_shape_name == other.explicit_shape_name
            shape_sizes_check = (self.profile.shape[0] == other.profile.shape[0] and self.profile.shape[1] == other.profile.shape[1])
            return shape_sizes_check and shape_type_check
        else:
            return False


# %% Overall utility functions
def force_precompilation():
    """
    Force compilation of computing functions for round and ellipse 'precise' shaped objects.

    Note that even this precompilation doesn't guarantee the acceleration of methods in UscopeScene class (use its '' method instead).

    Returns
    -------
    None.

    """
    if numba_installed:
        probe_obj = FluorObj(typical_size=2.0); probe_obj.get_shape(accelerated=True); del probe_obj  # for round shape
        probe_obj = FluorObj(shape_type='ellipse', typical_size=(2.65, 2.0, np.pi/6.0)); probe_obj.get_shape(accelerated=True); del probe_obj
    else:
        __warn_message = "Acceleration isn't possible because 'numba' library not installed in the current environment"
        warnings.warn(__warn_message)

# %% Define default export classes and methods used with import * statement (import * from fluoscenepy)
__all__ = ['UscopeScene', 'FluorObj', 'force_precompilation']

# %% Tests of different scenarios
if __name__ == "__main__":
    plt.close("all"); test_computed_centered_beads = False; test_precise_centered_bead = False; test_computed_shifted_beads = False
    test_precise_shifted_beads = False; test_ellipse_centered = False; test_ellipse_shifted = False; test_casting = False
    test_cropped_shapes = False; test_put_objects = False; test_generate_objects = False; test_overall_gen = False
    test_precise_shape_gen = False; test_round_shaped_gen = False; test_adding_noise = False; test_various_noises = False
    shifts = (-0.2, 0.44); test_cropping_shifted_circles = False; shifts1 = (0.0, 0.0); shifts2 = (-0.14, 0.95); shifts3 = (0.875, -0.99)
    test_compiling_acceleration = False  # testing the acceleration through compilation using numba
    test_placing_circles = False  # testing speed up placing algorithm
    prepare_centered_docs_images = False  # for making centered sample images for preparing Readme file about this project
    prepare_shifted_docs_images = False; shifts_sample = (0.24, 0.0)  # for making shifted sample images for preparing Readme
    prepare_scene_samples = False  # for preparing illustrative scenes with placed on them objects
    prepare_favicon_img = False; prepare_large_favicon_img = False  # generate picture with dense round objects

    # Check if skimage is installed and set the flag for using it for saving the raw generated images below
    # Although skimage is not included in the dependencies of the package
    saving_possible = True
    try:
        from skimage import io
    except ModuleNotFoundError:
        saving_possible = False

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
        gb8 = FluorObj(typical_size=2.0); gb8.get_shape(); gb8.crop_shape(); gb8.plot_shape()
        gb14 = FluorObj(typical_size=4.75); gb14.get_shape(); gb14.crop_shape(); gb14.plot_shape()
    # Testing cropping of shifted circles with precise borders
    if test_cropping_shifted_circles:
        gb16 = FluorObj(typical_size=3.75, center_shifts=shifts1); gb16.get_shape(); gb16.plot_shape()
        gb16.get_casted_shape(max_pixel_value=201); gb16.crop_shape(); gb16.plot_shape(); gb16.plot_cast_shape()
        gb17 = FluorObj(typical_size=3.75, center_shifts=shifts2); gb17.get_shape(); gb17.crop_shape(); gb17.plot_shape()
        gb18 = FluorObj(typical_size=3.75, center_shifts=shifts3); gb18.get_shape(); gb18.plot_shape(); gb18.crop_shape(); gb18.plot_shape()
        gb20 = FluorObj(typical_size=3.75, center_shifts=shifts); gb20.get_shape(); gb20.plot_shape()
        gb20.get_casted_shape(max_pixel_value=2068, image_type='uint16'); gb20.crop_shape(); gb20.plot_cast_shape()
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
    if test_precise_shifted_beads:
        gb8 = FluorObj(typical_size=2.0, center_shifts=shifts); gb8.get_shape(); gb8.crop_shape(); gb8.plot_shape()
        if test_casting:
            gb8.get_casted_shape(255, 'uint8'); gb8.plot_cast_shape()
    # Test of ellipse shaped objects generation
    if test_ellipse_centered:
        gb11 = FluorObj(shape_type='ellipse', typical_size=(4.4, 2.5, np.pi/3)); gb11.get_shape(); gb11.plot_shape()
        if test_casting:
            gb11.get_casted_shape(255, 'uint8'); gb11.plot_cast_shape()
    if test_ellipse_shifted:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.crop_shape(); gb12.plot_shape()
        if test_casting:
            gb12.get_casted_shape(255, 'uint8'); gb12.plot_cast_shape()
    # Testing cropping of shifted ellipse
    if test_cropped_shapes:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.plot_shape()
    # Testing of putting the manually generated objects on the scene
    if test_put_objects:
        scene = UscopeScene(width=24, height=21); gb8 = FluorObj(typical_size=2.78, center_shifts=shifts); gb8.get_shape()
        gb12 = FluorObj(shape_type='el', typical_size=(4.9, 2.8, np.pi/3)); gb12.get_shape(); gb12.get_casted_shape(max_pixel_value=255)
        gb8.get_casted_shape(max_pixel_value=254); gb8.crop_shape(); gb12.crop_shape()  # gb8.plot_shape(); gb12.plot_shape()
        gb8.set_image_sizes(scene.shape); gb12.set_image_sizes(scene.shape); gb8.set_coordinates((-1, 3)); gb12.set_coordinates((12, 5))
        gb9 = FluorObj(typical_size=3.0, center_shifts=(0.25, -0.1)); gb9.get_shape(); gb9.get_casted_shape(max_pixel_value=251)
        gb9.crop_shape(); gb9.set_image_sizes(scene.shape); gb9.set_coordinates((10, 18))
        scene.put_objects_on(fluo_objects=(gb8, gb12)); scene.put_objects_on(fluo_objects=(gb9, )); scene.show_scene()
    # Testing of generating scenes with various settings
    if test_generate_objects:
        objs = UscopeScene.get_random_objects(mean_size=4.2, size_std=1.5, shapes='r', intensity_range=(230, 252), n_objects=2)
        objs2 = UscopeScene.get_random_objects(mean_size=(7.3, 5), size_std=(2, 1.19), shapes='el',
                                               intensity_range=(220, 250), n_objects=2)
        scene = UscopeScene(width=45, height=38); objs = scene.set_random_places(objs); objs2 = scene.set_random_places(objs2)
        scene.put_objects_on(objs); scene.put_objects_on(objs2, save_objects=False); scene.show_scene()
        # scene2 = UscopeScene(width=55, height=46); scene2.show_scene()  # will show 'Blank' image
    if test_overall_gen:
        objs3 = UscopeScene.get_random_objects(mean_size=(8.3, 5.4), size_std=(2, 1.19),
                                               shapes='mixed', intensity_range=(182, 250), n_objects=5)
        scene = UscopeScene(width=55, height=46); scene.spread_objects_on(objs3); scene.show_scene(color_map='gray')
    if test_precise_shape_gen:
        objs4 = UscopeScene.get_random_objects(mean_size=(6.21, 5.36), size_std=(1.25, 0.95), shapes='mixed', intensity_range=(182, 250),
                                               n_objects=5, verbose_info=True)
        scene = UscopeScene(width=32, height=28); scene2 = UscopeScene(width=32, height=28)
        objs4_placed = scene.set_random_places(objs4, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene.put_objects_on(objs4, save_only_objects_inside=True); scene.show_scene()
        objs4_placed2 = scene2.set_random_places(objs4, overlapping=True, touching=False, only_within_scene=True, verbose_info=True)
        scene2.put_objects_on(objs4, save_only_objects_inside=True); scene2.show_scene()
    if test_round_shaped_gen:
        objs5 = UscopeScene.get_round_objects(mean_size=7.5, size_std=1.2, intensity_range=(188, 251), n_objects=55)
        scene3 = UscopeScene(width=102, height=90)
        objs5_pl = scene3.set_random_places(objs5, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene3.put_objects_on(objs5, save_only_objects_inside=True); scene3.show_scene()
    if test_adding_noise:
        objs6 = UscopeScene.get_random_objects(mean_size=(8.11, 6.36), size_std=(1.05, 0.92), shapes='mixed', intensity_range=(180, 245),
                                               n_objects=4, verbose_info=True)
        scene = UscopeScene(width=48, height=41)
        objs6_placed = scene.set_random_places(objs6, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene.put_objects_on(objs6_placed, save_only_objects_inside=True); scene.show_scene(str_id="Without additional noise")
        scene.add_noise(); scene.show_scene(str_id="With Poisson  & Gaussian noise")
    # Test adding noise to float image and various mean / sigma values
    if test_various_noises:
        objs6 = UscopeScene.get_random_objects(mean_size=(8.61, 6.36), size_std=(1.05, 0.92), shapes='mixed', intensity_range=(180, 245),
                                               n_objects=4, verbose_info=True)
        scene8 = UscopeScene(width=48, height=37)
        obj61_p = scene8.set_random_places(objs6, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene8.put_objects_on(obj61_p, save_only_objects_inside=True); scene8.show_scene("Without additional noise")
        scene8.add_noise(); scene8.show_scene("With default noise")
        scene8.add_noise(mean_noise=180//6, sigma_noise=180//9); scene8.show_scene("With stronger noise")
        scene8.add_noise(mean_noise=180//4, sigma_noise=180//6); scene8.show_scene("With much more stronger noise")
        scene8.remove_noise(); scene8.show_scene("Removed noise")
    # Test reworked algorithm for placing large number of circles that should / not lay within the scene, not touching / overlapping
    if test_placing_circles:
        objs12 = UscopeScene.get_round_objects(mean_size=8.75, size_std=1.4, intensity_range=(2000, 4000), n_objects=21, image_type=np.uint16)
        scene12 = UscopeScene(width=92, height=84, image_type='uint16'); scene14 = UscopeScene(width=92, height=84, image_type='uint16')
        objs12_placed = scene12.set_random_places(objs12, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene12.put_objects_on(objs12_placed, save_only_objects_inside=True); scene12.show_scene("Circles within scene")
        objs12_pl2 = scene14.set_random_places(objs12, overlapping=False, touching=False, only_within_scene=False, verbose_info=True)
        scene14.put_objects_on(objs12_pl2, save_only_objects_inside=True); scene14.show_scene("Circles partially within scene")
        scene15 = UscopeScene(width=41, height=41, image_type='uint16'); objs12_pl3 = scene14.set_random_places(objs12, verbose_info=True)
        scene15.put_objects_on(objs12_pl3, save_only_objects_inside=True); scene15.show_scene("Circles default parameters")
    # Preparing images for the documentation (README in the project repo)
    if prepare_centered_docs_images:
        objs1 = FluorObj(typical_size=2.0, border_type="computed", shape_method="oversampled circle")
        objs2 = FluorObj(typical_size=2.0, border_type="computed", shape_method="circle")
        objs3 = FluorObj(typical_size=2.0, border_type="computed", shape_method="undersampled circle")
        objs1.get_shape(); objs1.plot_shape(); objs2.get_shape(); objs2.plot_shape(); objs3.get_shape(); objs3.plot_shape()
        objs4 = FluorObj(typical_size=2.0); objs4.get_shape(); objs4.plot_shape()
        objs5 = FluorObj(typical_size=4.8); objs5.get_shape(); objs5.plot_shape(); objs5.get_shaping_functions()
        objs6 = FluorObj(typical_size=4.8, border_type="co", shape_method="bump3"); objs6.get_shape(); objs6.plot_shape()
    if prepare_shifted_docs_images:
        objs7 = FluorObj(typical_size=2.0, center_shifts=shifts_sample, border_type="computed", shape_method="circle")
        objs8 = FluorObj(typical_size=2.0, center_shifts=shifts_sample)
        objs7.get_shape(); objs7.plot_shape(); objs8.get_shape(); objs8.plot_shape()
        objel3 = FluorObj(shape_type='ellipse', typical_size=(4.8, 3.3, np.pi/6), center_shifts=shifts_sample)
        objel3.get_shape(); objel3.plot_shape()
    if prepare_scene_samples:
        force_precompilation()  # forcing precompilation by numba
        samples1 = UscopeScene.get_random_objects(mean_size=(9.5, 8.0), size_std=(1.15, 0.82), shapes='mixed', intensity_range=(186, 254),
                                                  n_objects=25, verbose_info=True, accelerated=True)
        scene_s = UscopeScene(width=104, height=92)
        samples1_pl = scene_s.set_random_places(samples1, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene_s.put_objects_on(samples1_pl, save_only_objects_inside=True); scene_s.show_scene("Scene without noise", color_map="gray")
        if saving_possible:
            io.imsave(Path.home().joinpath("Scene_without_noise.png"), scene_s.image)
        scene_s.add_noise(); scene_s.show_scene("Scene with added noise (default parameters)", color_map="gray")
        if saving_possible:
            io.imsave(Path.home().joinpath("Scene_with_noise.png"), scene_s.image)
    # Testing acceleration by using numba compilation in the imported module
    if test_compiling_acceleration:
        force_precompilation()  # force precompilation of functions by numba
        # Repeat checks twice for repeatability
        for i in range(2):
            # Computing discrete round shape with the attempt to accelerate computation by using numba compilation in the module
            t_ov_1 = time.perf_counter(); objs10 = FluorObj(typical_size=12.0); objs10.get_shape(accelerated=True); objs10.plot_shape()
            elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
            if elapsed_time_ov > 1000:
                elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
                print(f"Compiled round shape computation took {elapsed_time_ov} seconds", flush=True)
            else:
                print(f"Compiled round shape computation took {elapsed_time_ov} milliseconds", flush=True)
            # Standard computing discrete round shape
            t_ov_1 = time.perf_counter(); objs11 = FluorObj(typical_size=12.0); objs11.get_shape(); objs11.plot_shape()
            elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
            if elapsed_time_ov > 1000:
                elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
                print(f"Standard round shape computation took {elapsed_time_ov} seconds", flush=True)
            else:
                print(f"Standard round shape computation took {elapsed_time_ov} milliseconds", flush=True)
            # Computing discrete ellipse shape with the attempt to accelerate computation by using numba compilation in the module
            t_ov_1 = time.perf_counter(); objs12 = FluorObj(shape_type='ellipse', typical_size=(7.5, 6.0, np.pi/3))
            objs12.get_shape(accelerated=True); objs12.plot_shape(); elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
            if elapsed_time_ov > 1000:
                elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
                print(f"Compiled ellipse shape computation took {elapsed_time_ov} seconds", flush=True)
            else:
                print(f"Compiled ellipse shape computation took {elapsed_time_ov} milliseconds", flush=True)
            # Standard computing discrete ellipse shape
            t_ov_1 = time.perf_counter(); objs13 = FluorObj(shape_type='ellipse', typical_size=(7.5, 6.0, np.pi/3))
            objs13.get_shape(); objs13.plot_shape(); elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
            if elapsed_time_ov > 1000:
                elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
                print(f"Standard ellipse shape computation took {elapsed_time_ov} seconds", flush=True)
            else:
                print(f"Standard ellipse shape computation took {elapsed_time_ov} milliseconds", flush=True)
            if i == 0:
                plt.close('all'); del objs10, objs11, objs12, objs13
    # Prepare sample images for using as favicons in the documentation and for the main documentation page
    if prepare_favicon_img:
        round_objs2 = UscopeScene.get_round_objects(mean_size=10.0, size_std=2.0, intensity_range=(242, 252), n_objects=100)
        scene_favicon = UscopeScene(width=256, height=256)
        round_objs2 = scene_favicon.set_random_places(round_objs2, overlapping=False, touching=False,
                                                      only_within_scene=False, verbose_info=True)
        scene_favicon.put_objects_on(round_objs2); scene_favicon.add_noise(); scene_favicon.show_scene(color_map='gray')
        if saving_possible:
            io.imsave(Path.home().joinpath("favicon.png"), scene_favicon.image)
    if prepare_large_favicon_img:
        round_objs3 = UscopeScene.get_round_objects(mean_size=8.0, size_std=1.0, intensity_range=(240, 252), n_objects=20)
        scene_favicon2 = UscopeScene(width=64, height=64)
        round_objs3 = scene_favicon2.set_random_places(round_objs3, overlapping=False, touching=False,
                                                       only_within_scene=False, verbose_info=True)
        scene_favicon2.put_objects_on(round_objs3); scene_favicon2.add_noise(); scene_favicon2.show_scene(color_map='cividis')
        custom_path = ""
        if len(custom_path) > 0 and saving_possible:
            io.imsave(Path(custom_path), scene_favicon2.image)
