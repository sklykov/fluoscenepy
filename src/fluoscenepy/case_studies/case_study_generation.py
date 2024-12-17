# -*- coding: utf-8 -*-
"""
Case studies of implemented methods usage.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Global imports
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt

# %% Local (package-scoped) imports
# Add the main script to the sys path for importing
root_dir = Path(__file__).parent.parent; main_script_path = str(root_dir.absolute())
if main_script_path not in sys.path:
    sys.path.append(main_script_path)
# Import script directly from added absolute path
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from fluoscene import UscopeScene, force_precompilation

# %% Parameters - flags for making studies
check_uncompiled_generation_performance = False  # checked
check_globally_precompiled_methods = False  # checked
check_compiled_method_class = False  # checked
check_placing_objects = True

# %% Script run
if __name__ == "__main__":
    uscene = UscopeScene(width=200, height=180, image_type=np.uint16)
    # Check unaccelerated small particles generation
    if check_uncompiled_generation_performance:
        objs = uscene.get_random_objects(mean_size=4.0, size_std=0.25, intensity_range=(250, 255), n_objects=20, verbose_info=True)
    # Check how precompilation of methods globally accelerates the shapes generation
    if check_globally_precompiled_methods:
        force_precompilation()
        objs = uscene.get_random_objects(mean_size=4.0, size_std=0.25, intensity_range=(250, 255), n_objects=30, verbose_info=True,
                                         accelerated=True)  # for round objects: ~ 295-305 ms calculation
    # Check class-binded accelerated methods
    if check_compiled_method_class:
        # objs = uscene.get_objects_acc(mean_size=2.0, size_std=0.05, intensity_range=(250, 255), n_objects=1, verbose_info=True)
        uscene.precompile_methods(verbose_info=True)
        # objs2 = uscene.get_objects_acc(mean_size=4.0, size_std=0.25, intensity_range=(250, 255), n_objects=20, verbose_info=True)
        objs3 = uscene.get_objects_acc(mean_size=(8.0, 5.5), size_std=(0.5, 0.25), intensity_range=(250, 255),
                                       n_objects=15, verbose_info=True, shapes='mixed')
    # Check various conditions and acceleration of objects placing on scene
    if check_placing_objects:
        objs4 = uscene.get_round_objects(mean_size=10.5, size_std=1.5, intensity_range=(250, 255), n_objects=10, image_type=np.uint16)
        placed_objs4 = uscene.set_random_places(objs4, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        uscene.put_objects_on(placed_objs4, save_only_objects_inside=True)
        plt.close('all'); uscene.show_scene()
