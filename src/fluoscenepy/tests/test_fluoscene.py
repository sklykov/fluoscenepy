# -*- coding: utf-8 -*-
"""
Tests of the main module methods.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

@author: Sergei Klykov
@licence: MIT

"""
# %% Imports
import numpy as np

if __name__ != "__main__":
    from ..fluoscene import UscopeScene, FluorObj
    from ..utils.comp_funcs import get_radius_gaussian, get_ellipse_sizes


# %% Tests
def test_scene_initialization():
    try:
        UscopeScene(width=1, height=6)
        assert False, "Wrong initialization (UscopeScene(width=1, height=6)) not thrown the error"
    except ValueError:
        pass
    try:
        UscopeScene(width=12, height=3)
        assert False, "Wrong initialization (UscopeScene(width=12, height=3)) not thrown the error"
    except ValueError:
        pass
    try:
        UscopeScene(width=4.5, height=6)
        assert False, "Wrong initialization (UscopeScene(width=4.5, height=6)) not thrown the error"
    except TypeError:
        pass
    try:
        UscopeScene(width=5, height=4, image_type='none')
        assert False, "Wrong initialization (UscopeScene(width=2, height=2, img_type='none')) not thrown the error"
    except ValueError:
        pass
    scene = UscopeScene(width=51, height=100, image_type='uint16'); scene2 = UscopeScene(width=31, height=101, image_type=np.uint16)
    assert scene.max_pixel_value == scene2.max_pixel_value, 'Wrong initialization for image types np.uint16 and "uint16" '
    scene = UscopeScene(width=51, height=100, image_type='float'); scene2 = UscopeScene(width=31, height=101, image_type=np.float64)
    assert abs(scene.max_pixel_value - scene2.max_pixel_value) < 1E-6, 'Wrong initialization for image types np.uint16 and "uint16" '


def test_fluorobj_initialization():
    # False parameters provided for initialization - tests should fail
    try:
        FluorObj(typical_size=-0.4)
        assert False, "Wrong initialization (FluorObj(typical_size=-0.4)) not thrown the error"
    except ValueError:
        pass
    try:
        FluorObj(typical_size=5, center_shifts=1.0)
        assert False, "Wrong initialization (FluorObj(typical_size=5, center_shifts=1.0) not thrown the error"
    except TypeError:
        pass
    try:
        FluorObj(typical_size=(4.0, 4.0, 0.25), center_shifts=(0.0, 0.1))
        assert False, "Wrong initialization FluorObj(typical_size=(4.0, 4.0, 0.25), center_shifts=(0.0, 0.1) not thrown the error"
    except TypeError:
        pass
    try:
        FluorObj(typical_size=(1.49, 1.0, 2.18), center_shifts=(0.0, 0.1), shape_type='el')
        assert False, ("Wrong initialization FluorObj(typical_size=(1.49, 1.0, 2.18), center_shifts=(0.0, 0.1), shape_type='el') "
                       + " - not thrown the error for typical sizes out of range")
    except ValueError:
        pass
    # Checking initialization logic
    flobj = FluorObj(typical_size=4.75); flobj.get_shape(); flobj.crop_shape()
    assert flobj.profile.shape[0] > 4 and flobj.profile.shape[1] > 4, f"Profile sizes out of the expected range (5, 5): {flobj.profile.shape}"
    flobj = FluorObj(typical_size=(4.2, 5.1, 0.25*np.pi), shape_type='el'); flobj.get_shape(accelerated=False)
    flobj.get_casted_shape(max_pixel_value=1921, image_type='uint16'); flobj.crop_shape(); flobj.crop_shape()
    assert flobj.profile.shape == flobj.casted_profile.shape, "Profile and casted profile shapes aren't equal or double cropping causes errors"
    flobj = FluorObj(typical_size=2.0); flobj.get_shape(accelerated=True)
    assert flobj.profile.shape[1] == 3 and flobj.profile.shape[0] == 3, ("Profile sizes out of the expected range (3, 3): "
                                                                         + f"{flobj.profile.shape}")
    flobj = FluorObj(typical_size=(3.2, 4.2, -0.51*np.pi), shape_type='el'); flobj.get_shape(accelerated=True); flobj.crop_shape()
    flobj.get_casted_shape(max_pixel_value=100.0, image_type=np.float64)
    assert np.max(flobj.casted_profile) >= 100.0, f"Profile not casted properly, max value: {np.max(flobj.profile)}"


def test_objects_generation():
    # Not accelerated generation testing
    circles = UscopeScene.get_round_objects(mean_size=8, size_std=1.5, intensity_range=(202, 253), n_objects=5)
    scene = UscopeScene(width=55, height=42, image_type='uint16')
    placed_circles = scene.set_random_places(circles, overlapping=False, touching=False, only_within_scene=True)
    scene.put_objects_on(placed_circles, save_only_objects_inside=True)
    assert len(placed_circles) <= len(circles), "Number of placed circles more than number of generated circles"
    precise_objs = UscopeScene.get_random_objects(mean_size=(3.81, 2.36), size_std=(0.46, 0.22), shapes='mixed',
                                                  intensity_range=(180, 251), n_objects=4)
    scene2 = UscopeScene(width=64, height=52)
    placed_objs = scene2.set_random_places(precise_objs, overlapping=False, touching=False, only_within_scene=True)
    scene2.put_objects_on(placed_objs, save_only_objects_inside=True)
    assert len(placed_objs) <= len(precise_objs), "Number of placed objects more than number of generated 'precise' objects"
    # Testing for found bug in ver. 0.0.2 - getting wrong sizes for samples
    n_tested_objs = 79
    robjs2 = UscopeScene.get_round_objects(mean_size=12, size_std=8, intensity_range=(230, 254), n_objects=n_tested_objs)
    assert len(robjs2) == n_tested_objs, f"Round object generation not creating {n_tested_objs} objects as expected"
    # Acceleration generation testing
    accelerated_method_called = False
    try:
        import numba; numba_not_installed = False
        if numba is not None and not numba_not_installed:
            scene2.precompile_methods()  # verbose call of precompilation
            objs3 = scene2.get_objects_acc(mean_size=(2.5, 1.5), size_std=(1.0, 0.65), intensity_range=(240, 255),
                                           n_objects=5, shapes='ellipse')
            assert len(objs3) == 5, f"Number of generation objects by accelerated method is less than 3: {len(objs3)}"
            placed_objs3 = scene2.set_random_places(objs3, overlapping=False, touching=False, only_within_scene=True)
            assert len(placed_objs3) > 0, "Problem with placing common objects, no generated objects placed"
            accelerated_method_called = True
    except (ModuleNotFoundError, ImportError):
        numba_not_installed = True
    if numba_not_installed:
        if accelerated_method_called:
            assert False, "Accelerated method called wrongly (should be called because 'numba' not installed"


def test_other_methods():
    scene4 = UscopeScene(width=63, height=57); objs4 = None
    try:
        import numba
        if numba is not None:
            objs4 = scene4.get_objects_acc(mean_size=(3.75, 3.0), size_std=(0.25, 0.19), intensity_range=(195, 235),
                                           n_objects=4, shapes='mixed')
    except (ModuleNotFoundError, ImportError):
        objs4 = scene4.get_random_objects(mean_size=(3.75, 3.0), size_std=(0.25, 0.19), intensity_range=(195, 235),
                                          n_objects=4, shapes='mixed')
    if objs4 is not None:
        placed_objs = scene4.set_random_places(objs4, overlapping=False, touching=False, only_within_scene=True)
        scene4.put_objects_on(placed_objs, save_only_objects_inside=True)
        noisy_img = UscopeScene.noise2image(scene4.image)
        assert noisy_img.shape == scene4.image.shape, "Shapes of input and output images for noise2image() method not equal"
        assert np.max(noisy_img) != np.max(scene4.image), "Max pixel values for noisy and source image should be different"
        noisy_img2 = UscopeScene.noise2image(scene4.image, mean_g=2, sigma_g=21, gain_p=0.5)
        assert np.max(noisy_img2) < np.max(scene4.image), "Gain hasn't reduced max pixel value in process of adding noise to an image"
    else:
        assert False, "Something wrong with 'get_random_objects' method (check 'test_other_methods' function)"


def test_radiuses_generation():
    for i in range(250):
        r = get_radius_gaussian(r=1.5, r_std=1.0, mean_size=1.5, size_std=1.0)
        assert r >= 0.5, f"Generated r < 0.5: {round(r, 3)}"
        a, b, angle = get_ellipse_sizes(mean_size=(3.0, 2.0), size_std=(2.0, 1.0))
        r_min = min(a, b); r_max = max(a, b)
        assert r_min > 1.0 and r_max > 1.5, f"Sizes for ellipse {a, b} less smallest values"


def test_cast_images():
    scene = UscopeScene(width=267, height=232, image_type=np.uint16)
    objs = scene.get_round_objects(mean_size=12, size_std=2, intensity_range=(0, 4094), n_objects=14, image_type=scene.img_type)
    objs = scene.set_random_places(objs); scene.put_objects_on(objs); scene.add_noise(mean_g=200); scene.show_scene()
    img_neg_norm = UscopeScene.cast_image(scene.image)
    assert np.min(img_neg_norm) == -1.0 and np.max(img_neg_norm) == 1.0, "Image casting ('cast_image') to range [-1.0, 1.0] has a problem"
    img_int8 = UscopeScene.cast_image(scene.image, option='int8'); int8min = np.iinfo(np.int8).min; int8max = np.iinfo(np.int8).max
    assert np.min(img_int8) >= int8min and np.max(img_int8) == int8max, "Image casting ('cast_image') to int8 range [-127, 127] has a problem"
    img_int16 = UscopeScene.cast_image(scene.image, option='int16'); int16max = np.iinfo(np.int16).max; int16min = np.iinfo(np.int16).min
    assert np.min(img_int16) >= int16min and np.max(img_int16) == int16max, (f"Image casting ('cast_image') to int16 range [-{int16max}, "
                                                                             + f"{int16max}] has a problem")
