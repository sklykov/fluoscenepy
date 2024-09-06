# -*- coding: utf-8 -*-
"""
Script with wrapped functions for profile generation.

Wrapping by using @jit decorators from numba library.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Imports
import numpy as np
from typing import Union

# %% Checking and import the numba library for speeding up the calculation
global numba_installed
try:
    from numba import njit
    numba_installed = True
except ModuleNotFoundError:
    numba_installed = False


# %% Utility functions (simple one for achieving accelartion by using numba library compilation)
@njit
def distance_f_acc(i_px: Union[int, float, np.ndarray], j_px: Union[int, float, np.ndarray], i_centre: Union[int, float],
                   j_centre: Union[int, float]) -> Union[float, np.ndarray]:
    """
    Calculate the distances for pixels accelerated by numba compilation.

    Parameters
    ----------
    i_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    j_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    i_centre : int | float
        Center of an image.
    j_centre : int | float
        Center of an image.

    Returns
    -------
    float or numpy.ndarray
        Distances between provided pixels and the center of an image.

    """
    return np.round(np.sqrt(np.power(i_px - i_centre, 2) + np.power(j_px - j_centre, 2)), 6)


# %% Shape functions definitions (wrapper @njit and @jit don't work for the function below)
def discrete_shaped_bead_acc(r: float, center_shifts: tuple) -> np.ndarray:
    """
    Calculate the 2D shape of bead with the border pixel intensities defined from the counting area of these pixels within circle radius.

    Parameters
    ----------
    r : float
        Radius of a bead.
    center_shifts : tuple
        Shifts on axis of the bead center.

    Returns
    -------
    img : numpy.ndarray
        2D normalized shape of the bead.

    """
    max_size = int(round(2.5*r, 0))  # default size for the profile
    x_shift, y_shift = center_shifts  # unpacking the shift of the object center
    if abs(y_shift) > 0.0 or abs(x_shift) > 0.0:
        max_size += 1
        if abs(y_shift) >= 0.4 or abs(x_shift) >= 0.4:
            max_size += 1
    if max_size % 2 == 0:
        max_size += 1
    img = np.zeros(dtype=np.float32, shape=(max_size, max_size))  # creating by default float image, normalized to 1.0 as the max intensity
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if y_shift >= 0.0:
        i_center = i_img_center + y_shift
    else:
        i_center = i_img_center + y_shift + 1.0
    if x_shift >= 0.0:
        j_center = j_img_center + x_shift
    else:
        j_center = j_img_center + x_shift + 1.0
    net_shift = round(0.5*np.sqrt(y_shift*y_shift + x_shift*x_shift), 6)  # calculation the net shift of the picture center
    # Calculating the intensity distribution pixelwise with strict definition
    q_rad = round(0.25*r, 6); size_subareas = 626  # number of subareas + 1 that can make defined number of steps, like np.linspace(0, 1, 11)
    single_point_value = 1.0/(size_subareas-1)  # single point non-zero value used below for finding number of points within circle border
    normalization = single_point_value*size_subareas*size_subareas  # normalization for summation of all points
    for i in range(max_size):
        for j in range(max_size):
            distance = distance_f_acc(i, j, i_center, j_center)  # distance from the center to the pixel
            pixel_value = 0.0  # meaning the intensity in the pixel defined on the rules below
            if distance < q_rad:  # The pixel lays completely inside of circle border
                pixel_value = 1.0  # entire pixel lays inside the circle
            elif q_rad <= distance <= r + net_shift + 1.0:  # The pixel is intersecting with the circle border
                stop_checking = False  # flag for quitting these calculations if the pixel is proven to lay completely inside the circle
                # First, sort out the pixels that lay completely within the circle, but the distance is more than quarter of R:
                if i < i_center:
                    i_corner = i - 0.5
                else:
                    i_corner = i + 0.5
                if j < j_center:
                    j_corner = j - 0.5
                else:
                    j_corner = j + 0.5
                # Below - distance to the most distant point of the pixel
                distance_corner = distance_f_acc(i_corner, j_corner, i_center, j_center)
                if distance_corner <= r:
                    pixel_value = 1.0; stop_checking = True
                # So, the pixel's borders can potentially are intersected by the circle,
                # calculate the estimated intersection area for pixel intensity
                if not stop_checking:
                    i_m = i - 0.5; j_m = j - 0.5; i_p = i + 0.5; j_p = j + 0.5
                    x_row = np.linspace(start=i_m, stop=i_p, num=size_subareas); y_col = np.linspace(start=j_m, stop=j_p, num=size_subareas)
                    coords = np.meshgrid(x_row, y_col); distances = distance_f_acc(coords[0], coords[1], i_center, j_center)
                    circle_arc_area1 = np.where(distances <= r, single_point_value, 0.0)  # assigning the non-zero for intersected mesh grid points
                    S1 = round(np.sum(circle_arc_area1)/normalization, 6)
                    if S1 > 1.0:
                        S1 = 1.0  # in the rare cases the integration sum can be more than 1.0 due to the limited precision of numerical integration
                    pixel_value = S1  # assigning the found square of the area laying inside a circle
            img[i, j] = pixel_value  # assign the computed intensity to the stored 2D profile
    return img


# %% Default exports from this module
__all__ = ['discrete_shaped_bead_acc']
