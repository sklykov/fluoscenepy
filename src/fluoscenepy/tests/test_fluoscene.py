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
    from ..fluoscene import UscopeScene


# %% Tests
def test_initialization():
    try:
        scene = UscopeScene(width=1, height=2)
        assert False, "Wrong initialization (UscopeScene(width=1, height=2)) not thrown the error"
    except ValueError:
        pass
    try:
        scene = UscopeScene(width=2, height=1)
        assert False, "Wrong initialization (UscopeScene(width=2, height=1)) not thrown the error"
    except ValueError:
        pass
    try:
        scene = UscopeScene(width=2, height=2, img_type='none')
        assert False, "Wrong initialization (UscopeScene(width=2, height=2, img_type='none')) not thrown the error"
    except ValueError:
        pass
    scene = UscopeScene(width=51, height=100, img_type='uint16'); scene2 = UscopeScene(width=31, height=101, img_type=np.uint16)
    assert scene.max_pixel_value == scene2.max_pixel_value, 'Wrong initialization for image types np.uint16 and "uint16" '
    scene = UscopeScene(width=51, height=100, img_type='float'); scene2 = UscopeScene(width=31, height=101, img_type=np.float64)
    assert abs(scene.max_pixel_value - scene2.max_pixel_value) < 1E-6, 'Wrong initialization for image types np.uint16 and "uint16" '
