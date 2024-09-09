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


# %% Utility functions for round shaped objects (simple one for achieving acceleration by using numba library compilation)
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


@njit
def helper_calculations_bead(r: float, center_shifts: tuple) -> tuple:
    """
    Shift calculations to this function for compilation by numba.

    Parameters
    ----------
    r : float
        Radius.
    center_shifts : tuple
        Pixel shifts of the center.

    Returns
    -------
    tuple
        Couple of parameters required by the calling function.

    """
    max_size = int(round(2.5*r, 0))  # default size for the profile
    x_shift, y_shift = center_shifts  # unpacking the shift of the object center
    if abs(y_shift) > 0.0 or abs(x_shift) > 0.0:
        max_size += 1
        if abs(y_shift) >= 0.4 or abs(x_shift) >= 0.4:
            max_size += 1
    if max_size % 2 == 0:
        max_size += 1
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
    q_rad = round(0.25*r, 6)  # parameter for checking if a pixel is completely inside a circle
    size_subareas = 626  # number of subareas + 1 that can make defined number of steps, like np.linspace(0, 1, 11)
    single_point_value = 1.0/(size_subareas-1)  # single point non-zero value used below for finding number of points within circle border
    normalization = single_point_value*size_subareas*size_subareas  # normalization for summation of all points
    return max_size, i_center, j_center, q_rad, net_shift, size_subareas, single_point_value, normalization


# %% Round shape function (wrapper @njit and @jit don't work for the function below, most likely because of numpy.meshgrid function)
def discrete_shaped_bead_acc(r: float, center_shifts: tuple) -> np.ndarray:
    """
    Calculate the 2D shape of bead with the border pixel intensities defined from the counting area of these pixels within circle radius.

    Function accelerated by using compiled computation functions called inside.

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
    max_size, i_center, j_center, q_rad, net_shift, size_subareas, single_point_value, normalization = helper_calculations_bead(r, center_shifts)
    img = np.zeros(dtype=np.float32, shape=(max_size, max_size))  # creating by default float image, normalized to 1.0 as the max intensity
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


# %% Utility functions for ellipse shaped objects (simple one for achieving acceleration by using numba library compilation)
@njit
def ellipse_equation_acc(i_px: Union[int, float, np.ndarray], j_px: Union[int, float, np.ndarray], i_centre: Union[int, float],
                         j_centre: Union[int, float], a: Union[int, float], b: Union[int, float], angle: Union[int, float]) -> float:
    """
    Calculate the ellipse equation ratio for the defining if the point (i_px, j_px) lays inside an ellipse accelerated by numba compilation.

    Parameters
    ----------
    i_px : Union[int, float, np.ndarray]
        Point (pixel) coordinate i.
    j_px : Union[int, float, np.ndarray]
        Point (pixel) coordinate j.
    i_centre : Union[int, float]
        i coordinate of the center of ellipse (center of mass).
    j_centre : Union[int, float]
        j coordinate of the center of ellipse (center of mass).
    a : Union[int, float]
        'a' ellipse axis in pixels.
    b : Union[int, float]
        'b' ellipse axis in pixels.
    angle : Union[int, float]
        angle between 'a' axis and 'X' axis.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Ellipse

    Returns
    -------
    float
        Calculated equation (ratio) for the point.

    """
    angle = -angle  # makes the angle assignment consistent with the counter-clockwise count
    # Source for the equations below: https://en.wikipedia.org/wiki/Ellipse
    a_h2 = 0.25*a*a; b_h2 = 0.25*b*b  # half of axis (width and height) are used in the equations
    X = (j_px - j_centre)*np.cos(angle) + (i_px - i_centre)*np.sin(angle)  # projected X coordinate
    Y = -(j_px - j_centre)*np.sin(angle) + (i_px - i_centre)*np.cos(angle)  # projected Y coordinate
    i_diff2 = np.power(Y, 2); j_diff2 = np.power(X, 2)  # the general equation for ellipse
    return np.round(j_diff2/a_h2 + i_diff2/b_h2, 6)  # analytical equation for the shifted ellipse


# %% Ellipse shape definition accelerated by using the compiled function for defining belonging to the ellipse
def discrete_shaped_ellipse_acc(sizes: tuple, angle: float, center_shifts: tuple) -> np.ndarray:
    """
    Calculate ellipse shape project on the pixels.

    Function accelerated by using compiled computation functions called inside.

    Parameters
    ----------
    sizes : tuple
        Sizes a, b for an ellipse.
    angle : float
        Angle (counter-clockwise) between b and X axes.
    center_shifts : tuple
        Pixelwise shifts of an ellipse center.

    Returns
    -------
    img : numpy.ndarray
        2D normalized shape of the bead.

    """
    a, b = sizes  # from the definition of an ellipse: length of 2 axis
    max_d = max(a, b)  # for defining the largest and smallest axis of an ellipse
    max_size = int(round(1.25*max_d, 0))  # default size for the profile
    x_shift, y_shift = center_shifts  # unpacking the shift of the object center
    if abs(y_shift) > 0.0 or abs(x_shift) > 0.0:
        max_size += 1
        if abs(y_shift) > 0.4 or abs(x_shift) > 0.4:
            max_size += 1
    if max_size % 2 == 0:
        max_size += 1
    img = np.zeros(dtype=np.float32, shape=(max_size, max_size))  # creating by default float image, normalized to 1.0 as the max intensity
    i_img_center = max_size // 2; j_img_center = max_size // 2
    i_center = i_img_center + y_shift; j_center = j_img_center + x_shift
    # Calculating the intensity distribution pixelwise with strict definition
    size_subareas = 626  # number of subareas + 1 that can make defined number of steps, like np.linspace(0, 1, 11)
    single_point_value = 1.0/(size_subareas-1)  # single point non-zero value used below for finding number of points within circle border
    normalization = single_point_value*size_subareas*size_subareas  # normalization for summation of all points
    for i in range(max_size):
        for j in range(max_size):
            distance = ellipse_equation_acc(i, j, i_center, j_center, a, b, angle)  # equation for the ellipse testing
            pixel_value = 0.0  # meaning the intensity in the pixel defined on the rules below
            if 0.0 <= distance < 4.0:  # The pixel is intersecting with the ellipse border
                stop_checking = False  # flag for quitting these calculations if the pixel is proven to lay completely inside the circle
                # First, sort out the pixels that lay completely within the ellipse border, but the distance is more than quarter:
                if i <= i_center:  # shift to the left
                    i_corner = i - 0.5
                else:
                    i_corner = i + 0.5
                if j < j_center:  # shift to the bottom
                    j_corner = j - 0.5
                else:
                    j_corner = j + 0.5
                # Below - distance to the most distant point of the pixel
                distance_corner = ellipse_equation_acc(i_corner, j_corner, i_center, j_center, a, b, angle)
                if distance_corner < 0.5:  # empirical value for estimation that the pixel is entirely inside an ellipse
                    pixel_value = 1.0; stop_checking = True
                # So, the pixel's borders can potentially are intersected by the circle, calculate the estimated intersection area for pixel intensity
                if not stop_checking:
                    i_m = i - 0.5; j_m = j - 0.5; i_p = i + 0.5; j_p = j + 0.5
                    x_row = np.linspace(start=i_m, stop=i_p, num=size_subareas); y_col = np.linspace(start=j_m, stop=j_p, num=size_subareas)
                    coords = np.meshgrid(x_row, y_col); distances = ellipse_equation_acc(coords[0], coords[1], i_center, j_center, a, b, angle)
                    circle_arc_area1 = np.where(distances <= 1.0, single_point_value, 0.0)  # assigning the non-zero for intersected grid points
                    S1 = round(np.sum(circle_arc_area1)/normalization, 6)
                    if S1 > 1.0:
                        S1 = 1.0  # in the rare cases the integration sum can be more than 1.0 due to the limited precision of numerical integration
                    pixel_value = S1  # assigning the found square of the area laying inside a circle
            img[i, j] = pixel_value  # assign the computed intensity to the stored 2D profile
    return img


# %% Default exports from this module
__all__ = ['discrete_shaped_bead_acc', 'discrete_shaped_ellipse_acc']
