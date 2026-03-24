# -*- coding: utf-8 -*-
"""
Test the imports from "fluoscenepy" package.

@author: Sergei Klykov, @year: 2024, @licence: MIT

"""
import warnings
import pytest

numba_installed = True
try:
    import numba
    try:
        print("numba version:", numba.__version__)  # just for some usage of numba package to suppress unused warning
    except AttributeError:
        pass
except ModuleNotFoundError:
    numba_installed = False


# Note for further development: pytest library is capable to automatically recognize the project following 'src'
# layout without installation in the environment, so the test below is passed even though the package not installed by pip
# Although, the pytest should be called from the root folder of the project
@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore any custom UserWarning, in test scenario - 'numba' not installed Warning
def test_initialization():
    """
    Test import and basic usage of already installed package.

    Returns
    -------
    None.

    """
    try:
        from fluoscenepy import __version__, FluorObj, UscopeScene
        minor_ver = int(__version__.split(".")[1])
        if minor_ver >= 1:
            from fluoscenepy import precompile_fluoscene
            if numba_installed:
                precompile_fluoscene()  # call for numba precompilation
        else:
            from fluoscenepy import force_precompilation
            if numba_installed:
                force_precompilation()  # call for numba precompilation
        # Basic package usage. Accelerated flag below will throw out UserWarning if numba isn't installed
        fl_obj = FluorObj(typical_size=2.0, center_shifts=(-0.21, 0.32)); fl_obj.get_shape(accelerated=True)
        scene = UscopeScene(width=14, height=12); obj_pl = scene.set_random_places(tuple([fl_obj]))
        scene.put_objects_on(obj_pl)

    except (ImportError, ModuleNotFoundError):
        __warn_message = "\nPackage 'fluoscenepy' not installed, this integration test cannot be fulfilled"
        warnings.warn(__warn_message)


if __name__ == "__main__":
    test_initialization()
