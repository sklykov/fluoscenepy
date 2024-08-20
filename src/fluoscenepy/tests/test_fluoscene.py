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
    flobj = FluorObj(typical_size=4.75); flobj.get_shape(); flobj.crop_shape()
    assert flobj.profile.shape[0] > 4 and flobj.profile.shape[1] > 4, f"Profile sizes out of the expected range (5, 5): {flobj.profile.shape}"
