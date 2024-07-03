# -*- coding: utf-8 -*-
"""
Raw script for various object profile generation.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from math import e
# from matplotlib.patches import Circle


# %% Raw Object generation
def distance_f(i_px, j_px, i_centre, j_centre):
    """
    Calculate the distances for pixels.

    Parameters
    ----------
    i_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    j_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    i_centre : int
        Center of an image.
    j_centre : int
        Center of an image.

    Returns
    -------
    float or numpy.ndarray
        Distances between provided pixels and the center of an image.

    """
    return np.round(np.sqrt(np.power(i_px - i_centre, 2) + np.power(j_px - j_centre, 2)), 6)


def make_sample(radius: float, center_shift: tuple, max_intensity=255, test_plots: bool = False) -> np.ndarray:
    if radius < 1.0:
        radius = 1.0
    max_size = 4*int(round(radius, 0)) + 1
    i_shift, j_shift = center_shift
    net_shift = round(0.5*np.sqrt(i_shift*i_shift + j_shift*j_shift), 6)
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if abs(i_shift) <= 1.0 and abs(j_shift) <= 1.0:
        i_center = i_img_center + i_shift; j_center = j_img_center + j_shift
    else:
        i_center = i_img_center; j_center = j_img_center
    print("Center of a bead:", i_center, j_center)
    # Define image type
    if isinstance(max_intensity, int):
        if max_intensity <= 255:
            img_type = 'uint8'
        else:
            img_type = 'uint16'
    elif isinstance(max_intensity, float):
        if max_intensity > 1.0:
            max_intensity = 1.0
            img_type = 'float'
    else:
        raise ValueError("Specify Max Intencity for image type according to uint8, uint16, float")
    img = np.zeros(dtype=img_type, shape=(max_size, max_size))
    # Below - difficult to calculate the precise intersection of the circle and pixels
    # points = []
    q_rad = round(0.25*radius, 6); size_subareas = 1001; normalization = 0.001*size_subareas*size_subareas
    for i in range(max_size):
        for j in range(max_size):
            distance = distance_f(i, j, i_center, j_center)
            pixel_value = 0.0  # meaning the intensity in the pixel

            # Discrete function
            # if distance < 0.5*radius:
            #     pixel_value = max_intensity
            # elif distance < radius:
            #     pixel_value = float(max_intensity)*np.exp(pow(0.5, 1.25) - np.power(distance/radius, 1.25))
            # else:
            #     pixel_value = float(max_intensity)*np.exp(pow(0.5, 2.5) - np.power(distance/radius, 2.5))

            # # Continiuous bump function - too scaled result
            # r_exceed = 0.499; power = 4
            # if distance < radius*(1.0 + r_exceed):
            #     x = distance/(radius + r_exceed)
            #     x_pow = pow(x, power); b_pow = pow(1.0 + r_exceed, power)
            #     pixel_value = e*np.exp(b_pow/(x_pow - b_pow))

            # Discontinuous
            # x = distance / radius; ots = np.exp(-1.0/np.power(6.0, 2))
            # if distance < radius:
            #     pixel_value = np.exp(-np.power(x, 2)/np.power(6.0, 2))
            # else:
            #     x_shift = pow(x, 4); x_c = pow(0.95, 4)
            #     pixel_value = ots*np.exp(x_c - x_shift)

            # The center of bead lays always within single pixel
            # oversize = round(radius + 1 + net_shift, 6); bump_f_power = 16
            if distance < q_rad:
                pixel_value = 1.0  # entire pixel lays inside the circle
            # elif distance <= oversize - 0.5:
            #     x = pow(distance, bump_f_power); b = pow(oversize, bump_f_power)
            #     pixel_value = e*np.exp(b/(x - b))

            # The scheme below - overcomplicated and requires better definition of intersections of pixels and circle curvature
            # Rough estimate of the potentially outside pixels - they should be checked for intersection with the circle

            elif q_rad <= distance <= radius + net_shift + 1.0:
                stop_checking = False  # flag for quitting this calculations
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
                distance_corner = distance_f(i_corner, j_corner, i_center, j_center)
                if distance_corner <= radius:
                    pixel_value = 1.0; stop_checking = True

                # So, the pixel's borders can potentially are intersected by the circle, calculate the estimated intersection area
                if not stop_checking:
                    i_m = i - 0.5; j_m = j - 0.5; i_p = i + 0.5; j_p = j + 0.5
                    # circle_arc_area = np.zeros(shape=(size_subareas, size_subareas))
                    # h_x = (i_p - i_m)/size_subareas; h_y = (j_p - j_m)/size_subareas
                    # x_row = np.round(np.arange(start=i_m, stop=i_p+h_x/2, step=h_x), 6)
                    # y_col = np.round(np.arange(start=j_m, stop=j_p+h_y/2, step=h_y), 6)
                    x_row = np.linspace(start=i_m, stop=i_p, num=size_subareas); y_col = np.linspace(start=j_m, stop=j_p, num=size_subareas)
                    # print(np.min(x_row), np.max(x_row), np.min(y_col), np.max(y_col))
                    coords = np.meshgrid(x_row, y_col); distances = distance_f(coords[0], coords[1], i_center, j_center)
                    circle_arc_area1 = np.where(distances <= radius, 0.001, 0.0)
                    # print(circle_arc_area1.shape)
                    if np.max(circle_arc_area1) > 0.0 and radius <= 2.0 and test_plots:
                        plt.figure(f"{i, j}"); plt.imshow(circle_arc_area1)
                    # print(np.max(circle_arc_area1), np.min(circle_arc_area1))
                    # for y in range(size_subareas):
                    #     for x in range(size_subareas):
                    #         i_c = i_m + y*((i_p - i_m)/size_subareas); j_c = j_m + x*((j_p - j_m)/size_subareas)
                    #         distance_px = distance_f(i_c, j_c, i_center, j_center)
                    #         if distance_px <= radius:
                    #             circle_arc_area[y, x] = 1.0
                    # S = round(np.sum(circle_arc_area)/np.sum(pixel_area), 6)
                    S1 = round(np.sum(circle_arc_area1)/normalization, 6)
                    if S1 > 1.0:
                        print(np.min(x_row), np.max(x_row), np.min(y_col), np.max(y_col))
                        print(circle_arc_area1.shape)
                        print("Overflowed value", S1, "sum of pixels inside of the intersection:", np.sum(circle_arc_area1), "norm.:", normalization)
                        if test_plots:
                            plt.figure(f"[{i, j}]"); plt.imshow(circle_arc_area1)
                        S1 = 1.0
                    print(f"Found ratio for the pixel [{i, j}]:", S1); pixel_value = S1
                    # print(f"Found ratio for the pixel [{i, j}]:", S, "diff for - vect. implementations:", round(abs(S-S1), 6))
                    # r_diff1 = round(r2 - np.power(i_m - i_center, 2), 6); r_diff2 = round(r2 - np.power(j_m - j_center, 2), 6)
                    # r_diff3 = round(r2 - np.power(i_p - i_center, 2), 6); r_diff4 = round(r2 - np.power(j_p - j_center, 2), 6)
                    # found_points = 0; this_pixel_points = []
                    # # calculation of the j index
                    # if r_diff1 > 0.0:
                    #     j1 = round(j_center - np.sqrt(r_diff1), 6); j2 = round(j_center + np.sqrt(r_diff1), 6)
                    #     if j1 > 0.0 and j1 <= j_p and j1 >= j_m:
                    #         point = (i_m, j1); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if j2 > 0.0 and j2 <= j_p and j2 >= j_m:
                    #         point = (i_m, j2); points.append(point); this_pixel_points.append(point); found_points += 1
                    # if r_diff3 > 0.0:
                    #     j1 = round(j_center - np.sqrt(r_diff3), 6); j2 = round(j_center + np.sqrt(r_diff3), 6)
                    #     if j1 > 0.0 and j1 <= j_p and j1 >= j_m:
                    #         point = (i_p, j1); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if j2 > 0.0 and j2 <= j_p and j2 >= j_m:
                    #         point = (i_p, j2); points.append(point); this_pixel_points.append(point); found_points += 1
                    # # calculation of the i index
                    # if r_diff2 > 0.0:
                    #     i1 = round(i_center - np.sqrt(r_diff2), 6); i2 = round(i_center + np.sqrt(r_diff2), 6)
                    #     if i1 > 0.0 and i1 <= i_p and i1 >= i_m:
                    #         point = (i1, j_m); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if i2 > 0.0 and i2 <= i_p and i2 >= i_m:
                    #         point = (i2, j_m); points.append(point); this_pixel_points.append(point); found_points += 1
                    # if r_diff4 > 0.0:
                    #     i1 = round(i_center - np.sqrt(r_diff4), 6); i2 = round(i_center + np.sqrt(r_diff4), 6)
                    #     if i1 > 0.0 and i1 <= i_p and i1 >= i_m:
                    #         point = (i1, j_p); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if i2 > 0.0 and i2 <= i_p and i2 >= i_m:
                    #         point = (i2, j_p); points.append(point); this_pixel_points.append(point); found_points += 1

                    # # Calculated intersected square
                    # if found_points == 2:
                    #     print(f"Found intersections for the pixel [{i, j}]:", this_pixel_points)
                    #     x1, y1 = this_pixel_points[0]; x2, y2 = this_pixel_points[1]; S = 0.0
                    #     # Define intersection type - too complex (triangle, trapezoid, etc.)
                    #     # A = (i_m, j_m); B = (i_m, j_p); C = (i_p, j_m); D = (i_p, j_p)
                    #     x_m = 0.5*(x1 + x2); y_m = 0.5*(y1 + y2)
                    # print("middle point:", x_m, y_m)
                    # distance_m = round(np.sqrt(np.power(x_m - i_center, 2) + np.power(y_m - j_center, 2)), 6)
                    # if distance_m > distance:
                    #     S = 1.0 - 0.5*(distance_m - distance)
                    # else:
                    #     S = 1.0 + 0.5*(distance_m - distance)
                    # pixel_value = S
                    # print(f"Found points for the single pixel [{i, j}]:", found_points)

            # Pixel value scaling according the the provided image type
            pixel_value *= float(max_intensity)
            # Pixel value conversion to the image type
            if pixel_value > 0.0:
                if 'uint' in img_type:
                    pixel_value = int(round(pixel_value, 0))
                img[i, j] = pixel_value
    # points = set(points)  # excluding repeated found in the loop coordinates
    # print("found # of points:", len(points), "\ncoordinates:", points)
    return img


# %% Radial profile testing for the object generation
def profile1(x, sigma: float = 1.0):
    return np.exp(-np.power(x, 2)/np.power(sigma, 2))


def profile2(x):
    y = np.zeros(shape=x.shape); ots = np.exp(-np.power(1.0, 2)/np.power(3*1.0, 2))
    for i, el in enumerate(x):
        # if el < 0.5:
        #     y[i] = 1.0
        # elif el < 1.0:
        #     y[i] = np.exp(pow(0.5, 1.25) - np.power(el, 1.25))
        # else:
        #     y[i] = np.exp(pow(0.5, 2.5) - np.power(el, 2.5))
        if el <= 1.0:
            y[i] = np.exp(-np.power(el, 2)/np.power(3*1.0, 2))
        else:
            xc = 1.0; el = pow(el, 8)
            y[i] = ots*np.exp(xc-el)
    return y


def profile3(x, gamma: float = 1.0):
    gamma2 = gamma*gamma
    return gamma2/(np.power(x, 2) + gamma2)


def profile4(x):
    return np.exp(x)/np.power(1.0 + np.exp(x), 2)


def profile5(x, b: float = 1.0):
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el2 = el*el; b2 = b*b
            y[i] = np.exp(b2/(el2 - b2))
    return y*e


def profile6(x, b: float = 1.0):
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el3 = el*el*el; b3 = b*b*b
            y[i] = np.exp(b3/(el3 - b3))
    return y*e


# Testing the bump function difference in the radial profiles
def bump_f(x: np.ndarray, b: float = 1.0, power: int = 2) -> np.ndarray:
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el_pow = pow(el, power); b_pow = pow(b, power)
            y[i] = e*np.exp(b_pow/(el_pow - b_pow))
    return y


# %% 2D shapes based on continuous functions
def continuous_shaped_bead(r: float, center_shifts: tuple, bead_type: str) -> np.ndarray:
    """
    Bead with the shape defined depending on the provided type by using the continuous functions (distributions).

    Parameters
    ----------
    r : float
        Radius of a bead.
    center_shifts : tuple
        Shifts on axis of the bead center.
    bead_type : str
        Type of function to be used for calculating the shape.

    Returns
    -------
    img : numpy.ndarray
        2D normalized shape of the bead.

    """
    if bead_type == 'gaussian' or bead_type == 'g':
        # sigma = np.sqrt(r)  # direct estimation of the sigma variable
        sigma = 2.0*r/2.355  # based on FHWM
        max_size = int(round(3.0*sigma, 0))
    elif bead_type == 'lorentzian' or bead_type == 'lor':
        gamma = r*0.8; max_size = int(round(2.5*r, 0))
    elif bead_type == 'derivative of logistic func.' or bead_type == 'dlogf':
        max_size = int(round(2.5*r, 0))
    elif (bead_type == 'bump square' or bead_type == 'bump2' or bead_type == 'bump cube' or bead_type == 'bump3'
          or bead_type == 'bump ^8' or bead_type == 'bump8'):
        max_size = int(round(2.0*r, 0))
    elif bead_type == 'smooth circle' or bead_type == 'smcir':
        max_size = int(round(2.5*r, 0))
    else:
        max_size = int(round(4.0*r, 0))
    x_shift, y_shift = center_shifts
    if abs(y_shift) > 0.0 or abs(x_shift) > 0.0:
        max_size += 1
        if abs(y_shift) >= 0.4 or abs(x_shift) >= 0.4:
            max_size += 1
    if max_size % 2 == 0:
        max_size += 1
    img = np.zeros(dtype=np.float32, shape=(max_size, max_size))  # crating by default float image, normalized to 1.0 as the max intensity
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if y_shift >= 0.0:
        i_center = i_img_center + y_shift
    else:
        i_center = i_img_center + y_shift + 1.0
    if x_shift >= 0.0:
        j_center = j_img_center + x_shift
    else:
        j_center = j_img_center + x_shift + 1.0
    # Calculating the intensity distribution pixelwise
    cutoff_radius = 1.4*r  # for limiting pixels that is calculated for continuous profile
    for i in range(max_size):
        for j in range(max_size):
            distance = distance_f(i, j, i_center, j_center)
            if bead_type == 'gaussian' or bead_type == 'g':
                if distance < cutoff_radius:
                    img[i, j] = np.exp(-np.power(distance, 2)/np.power(sigma, 2))
            elif bead_type == 'lorentzian' or bead_type == 'lor':
                if distance < cutoff_radius:
                    img[i, j] = gamma/(np.power(distance, 2) + gamma)
            elif bead_type == 'derivative of logistic func.' or bead_type == 'dlogf':
                if distance < cutoff_radius:
                    img[i, j] = np.exp(distance/r)/np.power(1.0 + np.exp(distance/r), 2)
            elif 'bump' in bead_type:
                cutoff_proportion = 1.4  # regulates the cutoff size of the bump function in the fraction of radius
                if bead_type == 'bump square' or bead_type == 'bump2':
                    bump_pow = 2
                elif bead_type == 'bump cube' or bead_type == 'bump3':
                    bump_pow = 3
                elif bead_type == 'bump ^8' or bead_type == 'bump8':
                    bump_pow = 8
                b_pow = np.power(cutoff_proportion, bump_pow)*np.power(r, bump_pow)
                distance_pow = np.power(distance, bump_pow); b = cutoff_proportion*r
                if distance < b:
                    img[i, j] = np.exp(b_pow/(distance_pow - b_pow))
            elif bead_type == 'smooth circle' or bead_type == 'smcir':
                if distance < 0.5*r:
                    img[i, j] = 1.0  # the pixel lays completely inside of the circle
                elif 0.5*r <= distance < 1.0*r:
                    diff_distance = distance/r - 0.5  # difference between distance to the pixel and half of the radius
                    img[i, j] = 1.0 - np.power(diff_distance, 3)
                elif 1.0*r <= distance < cutoff_radius:
                    diff_distance = distance/r - 0.5  # difference between distance to the pixel and half of the radius
                    img[i, j] = 0.94 - diff_distance
    # Normalization of profile
    if (bead_type == 'derivative of logistic func.' or bead_type == 'dlogf' or bead_type == 'bump square' or bead_type == 'bump2'
       or bead_type == 'bump cube' or bead_type == 'bump3' or bead_type == 'bump ^8' or bead_type == 'bump8'):
        img /= np.max(img)
    return img


def discrete_shaped_bead(r: float, center_shifts: tuple) -> np.ndarray:
    """
    Calculate the 2D shape of bead with the border pixel intencities defined from the counting area of these pixels within circle radius.

    Parameters
    ----------
    r : float
        Radius of a bead.
    center_shifts : tuple
        Shifts on axis of the bead center.
    bead_type : str
        Type of function to be used for calculating the shape.

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
    img = np.zeros(dtype=np.float32, shape=(max_size, max_size))  # crating by default float image, normalized to 1.0 as the max intensity
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if y_shift >= 0.0:
        i_center = i_img_center + y_shift
    else:
        i_center = i_img_center + y_shift + 1.0
    if x_shift >= 0.0:
        j_center = j_img_center + x_shift
    else:
        j_center = j_img_center + x_shift + 1.0
    net_shift = round(0.5*np.sqrt(y_shift*y_shift + x_shift*x_shift), 6)  # calculation the shift of the picture center
    # Calculating the intensity distribution pixelwise with strict definition
    q_rad = round(0.25*r, 6); size_subareas = 626  # number of subareas + 1 that can make defined number of steps, like np.linspace(0, 1, 11)
    single_point_value = 1.0/(size_subareas-1)  # single point non-zero value used below for finding number of points within circle border
    normalization = single_point_value*size_subareas*size_subareas  # normalization for summation of all points
    for i in range(max_size):
        for j in range(max_size):
            distance = distance_f(i, j, i_center, j_center)  # distance from the center to the pixel
            pixel_value = 0.0  # meaning the intensity in the pixel defined on the rules below
            if distance < q_rad:  # The pixel lays completely inside of circle border
                pixel_value = 1.0  # entire pixel lays inside the circle
            elif q_rad <= distance <= r + net_shift + 1.0:  # The pixel is intersecting with the circle border
                stop_checking = False  # flag for quitting this calculations if the pixel is proven to lay completely inside of the circle
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
                distance_corner = distance_f(i_corner, j_corner, i_center, j_center)
                if distance_corner <= r:
                    pixel_value = 1.0; stop_checking = True
                # So, the pixel's borders can potentially are intersected by the circle, calculate the estimated intersection area for pixel intensity
                if not stop_checking:
                    i_m = i - 0.5; j_m = j - 0.5; i_p = i + 0.5; j_p = j + 0.5
                    x_row = np.linspace(start=i_m, stop=i_p, num=size_subareas); y_col = np.linspace(start=j_m, stop=j_p, num=size_subareas)
                    coords = np.meshgrid(x_row, y_col); distances = distance_f(coords[0], coords[1], i_center, j_center)
                    circle_arc_area1 = np.where(distances <= r, single_point_value, 0.0)  # assigning the non-zero number for intersected mesh grid points
                    S1 = round(np.sum(circle_arc_area1)/normalization, 6)
                    if S1 > 1.0:
                        S1 = 1.0  # in the rare cases the integration sum can be more than 1.0 due to the limited precision of numerical integration
                    pixel_value = S1  # assigning the found square of the area laying inside of a circle
            img[i, j] = pixel_value  # assign the computed intensity to the stored 2D profile
    return img


# %% Default exports from this module
__all__ = ['continuous_shaped_bead', 'discrete_shaped_bead']

# %% Tests
if __name__ == "__main__":
    test_disk_show = True; figsizes = (6.5, 6.5)
    # Testing disk representation
    if test_disk_show:
        i_shift = 0.23; j_shift = -0.591; disk_r = 6.0
        # disk1 = make_sample(radius=disk_r, center_shift=(i_shift, j_shift), test_plots=False)
        # plt.figure(figsize=figsizes); axes_img = plt.imshow(disk1, cmap=plt.cm.viridis); plt.tight_layout()
        # m_center, n_center = disk1.shape; m_center = m_center // 2 + i_shift; n_center = n_center // 2 + j_shift
        # axes_img.axes.add_patch(Circle((n_center, m_center), disk_r, edgecolor='red', facecolor='none'))
        r = np.arange(start=0.0, stop=1.3, step=0.02)
        profile1_f = profile1(r, 1.0)  # normal gaussian
        profile2_f = profile2(r)  # discontinuous function
        profile3_f = profile3(r, 1.0)  # lorentzian
        profile4_f = profile4(r); max_4 = np.max(profile4_f); profile4_f /= max_4  # derivative of logistic function
        profile5_f = profile5(r, 1.4)  # bump function
        profile6_f = profile6(r, 1.4)  # modified bump function
        plt.figure("Profiles Comparison"); plt.plot(r, profile1_f, r, profile2_f, r, profile3_f, r, profile4_f, r, profile5_f, r, profile6_f)
        plt.legend(['gaussian', 'discontinuous', 'lorentzian', 'd(logist.f)/dr', 'bump', 'mod. bump'])
        # Comparison of bump functions depending on the power of arguments
        size = 1.5; step_r = 0.01; r1 = np.arange(start=0.0, stop=size+step_r, step=step_r)
        bump2 = bump_f(r1, size, 2); bump4 = bump_f(r1, size, 4); bump3 = bump_f(r1, size, 3); bump64 = bump_f(r1, size, 64)
        bump8 = bump_f(r1, size, 8); bump16 = bump_f(r1, size, 16); bump32 = bump_f(r1, size, 32)
        plt.figure("Bump() Comparison"); plt.plot(r1, bump2, r1, bump3, r1, bump4, r1, bump8, r1, bump16)
        plt.legend(['^2', '^3', '^4', '^8', '^16']); plt.axvline(x=0.5); plt.axvline(x=1.0)
