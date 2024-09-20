# -*- coding: utf-8 -*-
"""
Test the imports from "fluoscenepy" package.

@author: Sergei Klykov, @year: 2024, @licence: MIT

"""
import warnings

numba_installed = True
try:
    import numba
    try:
        print(numba.__version__)  # just for some usage of numba package to suppress unused warning
    except AttributeError:
        pass
except ModuleNotFoundError:
    numba_installed = False


# Note for further development: pytest library is capable to automatically recognize the project following 'src'
# layout without installation in the environment, so the test below is passed even though the package not installed by pip
# Although, the pytest should be called from the root folder of the project
def test_initialization():
    try:
        from fluoscenepy import force_precompilation, FluorObj, UscopeScene

        if numba_installed:
            force_precompilation()  # call for numba precompilation

        # Basic package usage
        fl_obj = FluorObj(typical_size=2.0); fl_obj.get_shape(accelerated=True)
        scene = UscopeScene(width=14, height=12); obj_pl = scene.set_random_places([fl_obj])
        scene.put_objects_on(obj_pl)

    except (ImportError, ModuleNotFoundError):
        __warn_message = "\nPackage 'fluoscenepy' not installed, this integration test cannot be fulfilled"
        warnings.warn(__warn_message)


if __name__ == "__main__":
    test_initialization()
