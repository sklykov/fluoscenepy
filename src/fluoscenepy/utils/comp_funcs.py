# -*- coding: utf-8 -*-
"""
Default exports from this module (utils).

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
    if radius < 0.5:
        radius += random.uniform(a=0.6-radius, b=0.6)
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
    # Checking generated a_r, b_r axes for consistency (min axis >= 0.5, max axis >= 1.0)
    if a_r < 0.5:
        a_r += random.uniform(0.6-a_r, 0.6)
    elif b_r < 0.5:
        b_r += random.uniform(0.6-b_r, 0.6)
    max_axis = max(a_r, b_r)
    if max_axis < 1.0:
        if a_r == max_axis:
            a_r += random.uniform(1.1-a_r, 1.1)
        else:
            b_r += random.uniform(1.1-b_r, 1.1)
    return (a_r, b_r, angle)


def print_out_elapsed_t(initial_timing: int, operation: str = "operation"):
    elapsed_time_ov = int(round(1000.0*(time.perf_counter() - initial_timing), 0))
    if elapsed_time_ov > 1000:
        elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 1)
        print(f"Overall {operation} took {elapsed_time_ov} seconds", flush=True)
    else:
        print(f"Overall {operation} took {elapsed_time_ov} milliseconds", flush=True)
