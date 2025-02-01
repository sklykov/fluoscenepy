# -*- coding: utf-8 -*-
"""
Computational utility functions for 'fluoscenepy'.

@author: Sergei Klykov

"""
# %% Global imports
import random
from typing import Union
import numpy as np
import time


# %% Function defs.
def get_random_shape_props(mean_size: Union[float, int, tuple], size_std: Union[float, int, tuple]) -> tuple:
    """
    Select object properties from the provided parameters.

    Note that the returned shape type selected between 'round' and 'ellipse'.

    Parameters
    ----------
    mean_size : Union[float, int, tuple]
        Property to select from.
    size_std : Union[float, int, tuple]
        Property to select from.

    Returns
    -------
    tuple
        Selected (shape_type, mean_size, size_std).

    """
    shape_type = random.choice(['round', 'ellipse'])  # select between two supported types of objects
    # Define radius and radius std for round particles from the provided tuples for the ellipses (random choice between a and b)
    if isinstance(mean_size, tuple):
        r = random.choice(mean_size)
    else:
        r = mean_size
    if isinstance(size_std, tuple):
        r_std = random.choice(size_std)
    else:
        r_std = size_std
    return (shape_type, r, r_std)


def get_random_central_shifts() -> tuple:
    """
    Generate and return subpixel shifts from the exact center [0.0, 0.0] of the profile.

    Returns
    -------
    tuple
        (i_shift, j_shift) - subpixel shifts.

    """
    # Random selection of central shifts for placement
    i_shift = round(random.random(), 3); j_shift = round(random.random(), 3)  # random generation of central pixel shifts
    # Checking that shifts are generated in the subpixel range and correcting it if not
    if i_shift >= 1.0:
        i_shift -= round(random.random(), 3)*0.25
    if j_shift >= 1.0:
        j_shift -= round(random.random(), 3)*0.25
    sign_i = random.choice([-1.0, 1.0]); sign_j = random.choice([-1.0, 1.0]); i_shift *= sign_i; j_shift *= sign_j  # random signs
    return (i_shift, j_shift)


def get_random_max_intensity(intensity_range: tuple) -> Union[float, int]:
    """
    Select the intensity from the provided min, max range.

    Parameters
    ----------
    intensity_range : tuple
        (min I, max I).

    Returns
    -------
    Union(float, int)
        Returning the maximum intensity of the profile.

    """
    min_intensity, max_intensity = intensity_range  # assuming that only 2 values provided, if not - will throw an Exception
    # Random selection of max intensity for the profile casting
    if isinstance(min_intensity, int) and isinstance(max_intensity, int):
        fl_intensity = random.randrange(min_intensity, max_intensity, 1)
    elif isinstance(min_intensity, float) and isinstance(max_intensity, float):
        fl_intensity = random.uniform(a=min_intensity, b=max_intensity)
    return fl_intensity


def get_radius_gaussian(r: Union[float, int, None], r_std: Union[float, int, None], mean_size: Union[float, int],
                        size_std: Union[float, int]) -> float:
    """
    Get the random Gaussian-distributed radius.

    Parameters
    ----------
    r : Union[float, int]
        Selected radius.
    r_std : Union[float, int]
        Selected radius STD.
    mean_size : Union[float, int]
        Directly provided parameter by the method call if r is None.
    size_std : Union[float, int]
        Directly provided parameter by the method call if r_std is None.

    Raises
    ------
    ValueError
        If the wrongly provided mean_size and size_std parameters.

    Returns
    -------
    float
        Random Gaussian-distributed radius.

    """
    if r is not None and r_std is not None:
        radius = random.gauss(mu=r, sigma=r_std)
    else:
        if isinstance(mean_size, tuple) or isinstance(size_std, tuple):
            raise ValueError("Provided tuple with sizes for round shaped object, there expected only single number size")
        radius = random.gauss(mu=mean_size, sigma=size_std)
    # Checking generated radius for consistency
    radius = abs(radius)  # Gaussian distribution -> negative values also generated
    if radius < 0.5:
        radius += random.uniform(a=1.0-radius, b=1.0)
    return radius


def get_ellipse_sizes(mean_size: tuple, size_std: tuple) -> tuple:
    """
    Get the randomly Gaussian distributed axes sizes a, b and uniformly distributed angle.

    Parameters
    ----------
    mean_size : tuple
        Provided by the method call for unpacking sizes.
    size_std : tuple
        Provided by the method call for unpacking STD of sizes.

    Returns
    -------
    tuple
        (a axis, b axis, angle) - parameters for ellipse generation.

    """
    a, b = mean_size; a_std, b_std = size_std  # unpacking tuples assuming 2 of sizes packed there
    a_r = random.gauss(mu=a, sigma=a_std); b_r = random.gauss(mu=b, sigma=b_std)
    angle = random.uniform(a=0.0, b=2.0*np.pi)  # get random orientation for an ellipse
    a_r = abs(a_r); b_r = abs(b_r)  # sizes must be > 0
    # Checking generated a_r, b_r axes for consistency (min axis >= 1.0, max axis >= 1.5)
    if a_r < 1.0:
        a_r += random.uniform(2.0-a_r, 2.0)
    if b_r < 1.0:
        b_r += random.uniform(2.0-b_r, 2.0)
    max_axis = max(a_r, b_r)
    if max_axis < 1.5:
        if a_r == max_axis:
            a_r += random.uniform(3.0-a_r, 3.0)
        else:
            b_r += random.uniform(3.0-b_r, 3.0)
    return (a_r, b_r, angle)


def print_out_elapsed_t(initial_timing: float, operation: str = "Operation"):
    """
    Print out the elapsed time in comparison to the initial counter (from time.perf_counter()) in ms or sec.

    Parameters
    ----------
    initial_timing : float
        Provided by time.perf_counter() timing counter.
    operation : str, optional
        Name of operation to be printed out. The default is "operation".

    Returns
    -------
    None.

    """
    elapsed_time_ov = int(round(1000.0*(time.perf_counter() - initial_timing), 0))
    if elapsed_time_ov > 1000:
        elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 1)
        print(f"{operation} took: {elapsed_time_ov} seconds", flush=True)
    else:
        print(f"{operation} took: {elapsed_time_ov} milliseconds", flush=True)


# @jit(nopython=False)  # this acceleration by numba library doesn't work, function used for shortening code from the main module
def delete_coordinates_from_list(coordinates_list: list, input_list: list) -> list:
    """
    Delete coordinates from the copy of an input list.

    Parameters
    ----------
    coordinates_list : list
        List with coordinates to be deleted.
    input_list: list
        List for looking for and deleting the provided coordinates.

    Returns
    -------
    list
        Copy of an input list with the deleted coordinates.

    """
    cleaned_list = input_list[:]  # copy the input list
    for del_coord in coordinates_list:
        try:
            cleaned_list.remove(del_coord)
        except ValueError:
            pass
    return cleaned_list


def set_binary_mask_coords_in_loop(binary_mask: np.ndarray, i_obj_start: int, i_obj_end: int, i_limit: int,
                                   j_obj_start: int, j_obj_end: int, j_limit: int) -> np.ndarray:
    """
    Set the binary mask for the provided object (profile of an object) coordinates.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary mask used for avoiding objects intersection.
    i_obj_start : int
        Start i index.
    i_obj_end : int
        Stop i index.
    i_limit : int
        Upper limit for i index.
    j_obj_start : int
        Start j index.
    j_obj_end : int
        Stop j index.
    j_limit : int
        Upper limit for j index.

    Returns
    -------
    binary_mask : numpy.ndarray
        Processed binary mask.

    """
    binary_mask = binary_mask.copy()  # automatically coping the income array and working on it
    for i in range(i_obj_start, i_obj_end):
        for j in range(j_obj_start, j_obj_end):
            if 0 <= i < i_limit and 0 <= j < j_limit:
                binary_mask[i, j] = 1
    return binary_mask
