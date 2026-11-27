# -*- coding: utf-8 -*-
"""
Smoke-test the package's public imports and basic scene generation.

@author: Sergei Klykov, @year: 2024, @license: MIT

"""


def test_package_smoke() -> None:
    """
    Import the public API and draw a bead with a known peak intensity.

    Returns
    -------
    None

    """
    from fluoscenepy import FluorObj, UscopeScene, __version__, precompile_fluoscene

    # Check that both elements carry the expected logic (non-empty version and callable function)
    assert __version__
    assert callable(precompile_fluoscene)

    # test of basic methods
    scene = UscopeScene(width=14, height=12, numba_precompile=False)
    bead = FluorObj(typical_size=2.0); peak_intensity = 120; bead.get_shape()
    bead.get_casted_shape(max_pixel_value=peak_intensity, image_type=scene.img_type)
    bead.set_image_sizes(scene.shape)
    assert bead.set_coordinates((3, 4))
    scene.put_objects_on((bead,))

    assert scene.image is not None
    assert scene.image.shape == (12, 14)
    assert scene.image.max() == peak_intensity
