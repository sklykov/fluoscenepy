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


# %% Tests
def test_scene_initialization():
    try:
        scene = UscopeScene(width=1, height=6)
        assert False, "Wrong initialization (UscopeScene(width=1, height=2)) not thrown the error"
    except ValueError:
        pass
    try:
        scene = UscopeScene(width=12, height=3)
        assert False, "Wrong initialization (UscopeScene(width=12, height=3)) not thrown the error"
    except ValueError:
        pass
    try:
        scene = UscopeScene(width=4.5, height=6)
        assert False, "Wrong initialization (UscopeScene(width=4.5, height=6)) not thrown the error"
    except TypeError:
        pass
    try:
        scene = UscopeScene(width=5, height=4, image_type='none')
        assert False, "Wrong initialization (UscopeScene(width=2, height=2, img_type='none')) not thrown the error"
    except ValueError:
        pass
    scene = UscopeScene(width=51, height=100, image_type='uint16'); scene2 = UscopeScene(width=31, height=101, image_type=np.uint16)
    assert scene.max_pixel_value == scene2.max_pixel_value, 'Wrong initialization for image types np.uint16 and "uint16" '
    scene = UscopeScene(width=51, height=100, image_type='float'); scene2 = UscopeScene(width=31, height=101, image_type=np.float64)
    assert abs(scene.max_pixel_value - scene2.max_pixel_value) < 1E-6, 'Wrong initialization for image types np.uint16 and "uint16" '


def test_fluorobj_initialization():
    try:
        flobj = FluorObj(typical_size=-0.4)
        assert False, "Wrong initialization (FluorObj(typical_size=-0.4)) not thrown the error"
    except ValueError:
        pass
    try:
        flobj = FluorObj(typical_size=5, center_shifts=1.0)
        assert False, "Wrong initialization (FluorObj(typical_size=5, center_shifts=1.0) not thrown the error"
    except TypeError:
        pass
    try:
        flobj = FluorObj(typical_size=(4.0, 4.0, 0.25), center_shifts=(0.0, 0.1))
        assert False, "Wrong initialization FluorObj(typical_size=(4.0, 4.0, 0.25), center_shifts=(0.0, 0.1) not thrown the error"
    except TypeError:
        pass
    flobj = FluorObj(typical_size=4.75); flobj.get_shape(); flobj.crop_shape()
    assert flobj.profile.shape[0] > 4 and flobj.profile.shape[1] > 4, f"Profile sizes out of the expected range (5, 5): {flobj.profile.shape}"
    flobj = FluorObj(typical_size=(4.2, 5.1, 0.25*np.pi), shape_type='el'); flobj.get_shape()
    flobj.get_casted_shape(max_pixel_value=1921, image_type='uint16'); flobj.crop_shape(); flobj.crop_shape()
    assert flobj.profile.shape == flobj.casted_profile.shape, "Profile and casted profile shapes aren't equa or double cropping causes errors"
    flobj = FluorObj(typical_size=2.0); flobj.get_shape(accelerated=True)
    assert flobj.profile.shape[1] == 3 and flobj.profile.shape[0] == 3, ("Profile sizes out of the expected range (3, 3): "
                                                                         + f"{flobj.profile.shape}")


def test_objects_generation():
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
