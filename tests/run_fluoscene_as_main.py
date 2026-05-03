# -*- coding: utf-8 -*-
"""
Test features of 'fluoscenepy' by importing of editable installation (pip install -e . - command for such installation).

@author: @sklykov

"""
# %% Imports and checking its validity
from pathlib import Path
import numpy as np
import matplotlib
import importlib
from contextlib import suppress
import time
import warnings

# Explicit backend assignment for matplotlib - for compatibility between running configurations in Spyder and PyCharm IDEs
with suppress(ImportError):
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Import dev version of a package
import fluoscenepy.fluoscene
importlib.reload(fluoscenepy.fluoscene)
from fluoscenepy.fluoscene import precompile_fluoscene, FluorObj, UscopeScene, clean_fluoscene_cache


# %% Actual tests
if __name__ == "__main__":
    plt.close("all"); test_computed_centered_beads = False; test_precise_centered_bead = False; test_computed_shifted_beads = False
    test_precise_shifted_beads = False; test_ellipse_centered = False; test_ellipse_shifted = False; test_casting = False
    test_cropped_shapes = True; test_put_objects = False; test_generate_objects = False; test_overall_gen = False
    test_precise_shape_gen = False; test_round_shaped_gen = False; test_adding_noise = False; test_various_noises = False
    shifts = (-0.2, 0.44); test_cropping_shifted_circles = False; shifts1 = (0.0, 0.0); shifts2 = (-0.14, 0.95); shifts3 = (0.875, -0.99)
    test_compiling_acceleration = False  # testing the acceleration through compilation using numba
    test_placing_circles = False  # testing speed up placing algorithm
    prepare_centered_docs_images = False  # for making centered sample images for preparing Readme file about this project
    prepare_shifted_docs_images = False; shifts_sample = (0.24, 0.0)  # for making shifted sample images for preparing Readme
    prepare_scene_samples = False  # for preparing illustrative scenes with placed on them objects
    prepare_favicon_img = False; prepare_large_favicon_img = False  # generate picture with dense round objects
    test_add_noise_ext_img = False  # testing of adding noise to an external image
    test_cleaning_compilation_cache = False  # can be checked if local numba cache cleaned
    test_cast = True; check_warning = True  # casting images from different source to different data types
    show_valuable_round_objs = True

    # Check if skimage (not included in the project's dependencies) is installed and set the flag for using it for saving generated images
    saving_possible = True
    try:
        from skimage import io
    except ModuleNotFoundError:
        saving_possible = False

    # Testing the centered round objects generation
    if test_computed_centered_beads:
        gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g'); gb1.get_shape(); gb1.plot_shape()
        gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor'); gb2.get_shape(); gb2.plot_shape()
        gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf'); gb3.get_shape(); gb3.plot_shape()
        gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2'); gb4.get_shape(); gb4.plot_shape()
        gb5 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump3'); gb5.get_shape(); gb5.plot_shape()
        gb6 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump8'); gb6.get_shape(); gb6.plot_shape()
        gb7 = FluorObj(typical_size=2.0, border_type='co', shape_method='smcir'); gb7.get_shape(); gb7.plot_shape()
        gb9 = FluorObj(typical_size=2.0, border_type='co', shape_method='ovcir'); gb9.get_shape(); gb9.plot_shape()
        gb10 = FluorObj(typical_size=2.0, border_type='co', shape_method='uncir'); gb10.get_shape(); gb10.plot_shape()
    if test_precise_centered_bead:
        gb8 = FluorObj(typical_size=2.0); gb8.get_shape(); gb8.crop_shape(); gb8.plot_shape()
        gb14 = FluorObj(typical_size=4.75); gb14.get_shape(); gb14.crop_shape(); gb14.plot_shape()
    # Testing cropping of shifted circles with precise borders
    if test_cropping_shifted_circles:
        gb16 = FluorObj(typical_size=3.75, center_shifts=shifts1); gb16.get_shape(); gb16.plot_shape()
        gb16.get_casted_shape(max_pixel_value=201); gb16.crop_shape(); gb16.plot_shape(); gb16.plot_cast_shape()
        gb17 = FluorObj(typical_size=3.75, center_shifts=shifts2); gb17.get_shape(); gb17.crop_shape(); gb17.plot_shape()
        gb18 = FluorObj(typical_size=3.75, center_shifts=shifts3); gb18.get_shape(); gb18.plot_shape(); gb18.crop_shape(); gb18.plot_shape()
        gb20 = FluorObj(typical_size=3.75, center_shifts=shifts); gb20.get_shape(); gb20.plot_shape()
        gb20.get_casted_shape(max_pixel_value=2068, image_type='uint16'); gb20.crop_shape(); gb20.plot_cast_shape()
    # Testing the shifted from the center objects generation
    if test_computed_shifted_beads and not test_computed_centered_beads:
        gb1 = FluorObj(typical_size=2.0, border_type='co', shape_method='g', center_shifts=shifts); gb1.get_shape(); gb1.plot_shape()
        gb2 = FluorObj(typical_size=2.0, border_type='co', shape_method='lor', center_shifts=shifts); gb2.get_shape(); gb2.plot_shape()
        gb3 = FluorObj(typical_size=2.0, border_type='co', shape_method='dlogf', center_shifts=shifts); gb3.get_shape(); gb3.plot_shape()
        gb4 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump2', center_shifts=shifts); gb4.get_shape(); gb4.plot_shape()
        gb5 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump3', center_shifts=shifts); gb5.get_shape(); gb5.plot_shape()
        gb6 = FluorObj(typical_size=2.0, border_type='co', shape_method='bump8', center_shifts=shifts); gb6.get_shape(); gb6.plot_shape()
        gb7 = FluorObj(typical_size=2.0, border_type='co', shape_method='smcir', center_shifts=shifts); gb7.get_shape(); gb7.plot_shape()
        gb9 = FluorObj(typical_size=2.0, border_type='co', shape_method='ovcir', center_shifts=shifts); gb9.get_shape(); gb9.plot_shape()
        gb10 = FluorObj(typical_size=2.0, border_type='co', shape_method='uncir', center_shifts=shifts); gb10.get_shape(); gb10.plot_shape()
    if test_precise_shifted_beads:
        gb8 = FluorObj(typical_size=2.0, center_shifts=shifts); gb8.get_shape(); gb8.crop_shape(); gb8.plot_shape()
        if test_casting:
            gb8.get_casted_shape(255); gb8.plot_cast_shape()
    # Test of ellipse shaped objects generation
    if test_ellipse_centered:
        gb11 = FluorObj(shape_type='ellipse', typical_size=(4.4, 2.5, np.pi/3)); gb11.get_shape(); gb11.plot_shape()
        if test_casting:
            gb11.get_casted_shape(255); gb11.plot_cast_shape()
    if test_ellipse_shifted:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.crop_shape(); gb12.plot_shape()
        if test_casting:
            gb12.get_casted_shape(255); gb12.plot_cast_shape()
    # Testing cropping of shifted ellipse
    if test_cropped_shapes:
        gb12 = FluorObj(shape_type='el', typical_size=(4.4, 2.5, np.pi/3)); gb12.get_shape(shifts); gb12.plot_shape()
    # Testing of putting the manually generated objects on the scene
    if test_put_objects:
        scene = UscopeScene(width=24, height=21); gb8 = FluorObj(typical_size=2.78, center_shifts=shifts); gb8.get_shape()
        gb12 = FluorObj(shape_type='el', typical_size=(4.9, 2.8, np.pi/3)); gb12.get_shape(); gb12.get_casted_shape(max_pixel_value=255)
        gb8.get_casted_shape(max_pixel_value=254); gb8.crop_shape(); gb12.crop_shape()  # gb8.plot_shape(); gb12.plot_shape()
        gb8.set_image_sizes(scene.shape); gb12.set_image_sizes(scene.shape); gb8.set_coordinates((-1, 3)); gb12.set_coordinates((12, 5))
        gb9 = FluorObj(typical_size=3.0, center_shifts=(0.25, -0.1)); gb9.get_shape(); gb9.get_casted_shape(max_pixel_value=251)
        gb9.crop_shape(); gb9.set_image_sizes(scene.shape); gb9.set_coordinates((10, 18))
        scene.put_objects_on(fluo_objects=(gb8, gb12)); scene.put_objects_on(fluo_objects=(gb9, )); scene.show_scene()
    # Testing of generating scenes with various settings
    if test_generate_objects:
        objs = UscopeScene.get_random_objects(mean_size=4.2, size_std=1.5, shapes='r', intensity_range=(230, 252))
        objs2 = UscopeScene.get_random_objects(mean_size=(7.3, 5), size_std=(2, 1.19), shapes='el', intensity_range=(220, 250))
        scene = UscopeScene(width=45, height=38); objs = scene.set_random_places(objs); objs2 = scene.set_random_places(objs2)
        scene.put_objects_on(objs); scene.put_objects_on(objs2, save_objects=False); scene.show_scene()
        # scene2 = UscopeScene(width=55, height=46); scene2.show_scene()  # will show 'Blank' image
    if test_overall_gen:
        objs3 = UscopeScene.get_random_objects(mean_size=(8.3, 5.4), size_std=(2, 1.19), shapes='mixed',
                                               intensity_range=(182, 250), n_objects=5)
        scene = UscopeScene(width=55, height=46); scene.spread_objects_on(objs3); scene.show_scene(color_map='gray')
    if test_precise_shape_gen:
        objs4 = UscopeScene.get_random_objects(mean_size=(6.21, 5.36), size_std=(1.25, 0.95), shapes='mixed', intensity_range=(182, 250),
                                               n_objects=5, verbose_info=True)
        scene = UscopeScene(width=32, height=28); scene2 = UscopeScene(width=32, height=28)
        objs4_placed = scene.set_random_places(objs4, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene.put_objects_on(objs4, save_only_objects_inside=True); scene.show_scene()
        objs4_placed2 = scene2.set_random_places(objs4, touching=False, only_within_scene=True, verbose_info=True)
        scene2.put_objects_on(objs4, save_only_objects_inside=True); scene2.show_scene()
    if test_round_shaped_gen:
        objs5 = UscopeScene.get_round_objects(mean_size=7.5, size_std=1.2, intensity_range=(188, 251), n_objects=55)
        scene3 = UscopeScene(width=102, height=90)
        objs5_pl = scene3.set_random_places(objs5, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene3.put_objects_on(objs5, save_only_objects_inside=True); scene3.show_scene()
    if test_adding_noise:
        objs6 = UscopeScene.get_random_objects(mean_size=(8.11, 6.36), size_std=(1.05, 0.92), shapes='mixed', intensity_range=(180, 245),
                                               n_objects=4, verbose_info=True)
        scene = UscopeScene(width=48, height=41)
        objs6_placed = scene.set_random_places(objs6, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene.put_objects_on(objs6_placed, save_only_objects_inside=True); scene.show_scene(str_id="Without additional noise")
        scene.add_noise(); scene.show_scene(str_id="With Poisson  & Gaussian noise")
    # Test adding noise to float image and various mean / sigma values
    if test_various_noises:
        objs6 = UscopeScene.get_random_objects(mean_size=(8.61, 6.36), size_std=(1.05, 0.92), shapes='mixed', intensity_range=(180, 245),
                                               n_objects=4, verbose_info=True)
        scene8 = UscopeScene(width=48, height=37)
        obj61_p = scene8.set_random_places(objs6, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene8.put_objects_on(obj61_p, save_only_objects_inside=True); scene8.show_scene("Without additional noise")
        scene8.add_noise(); scene8.show_scene("With default noise")
        scene8.add_noise(mean_g=180//6, sigma_g=180//9); scene8.show_scene("With stronger noise")
        scene8.add_noise(mean_g=180//4, sigma_g=180//6); scene8.show_scene("With much more stronger noise")
        scene8.remove_noise(); scene8.show_scene("Removed noise")
    # Test reworked algorithm for placing large number of circles that should / not lay within the scene, not touching / overlapping
    if test_placing_circles:
        objs12 = UscopeScene.get_round_objects(mean_size=8.75, size_std=1.4, intensity_range=(2000, 4000),
                                               n_objects=21, image_type=np.uint16)
        scene12 = UscopeScene(width=92, height=84, image_type='uint16'); scene14 = UscopeScene(width=92, height=84, image_type='uint16')
        objs12_placed = scene12.set_random_places(objs12, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene12.put_objects_on(objs12_placed, save_only_objects_inside=True); scene12.show_scene("Circles within scene")
        objs12_pl2 = scene14.set_random_places(objs12, overlapping=False, touching=False, verbose_info=True)
        scene14.put_objects_on(objs12_pl2, save_only_objects_inside=True); scene14.show_scene("Circles partially within scene")
        scene15 = UscopeScene(width=41, height=41, image_type='uint16'); objs12_pl3 = scene14.set_random_places(objs12, verbose_info=True)
        scene15.put_objects_on(objs12_pl3, save_only_objects_inside=True); scene15.show_scene("Circles default parameters")
    # Preparing images for the documentation (README in the project repo)
    if prepare_centered_docs_images:
        objs1 = FluorObj(typical_size=2.0, border_type="computed", shape_method="oversampled circle")
        objs2 = FluorObj(typical_size=2.0, border_type="computed", shape_method="circle")
        objs3 = FluorObj(typical_size=2.0, border_type="computed", shape_method="undersampled circle")
        objs1.get_shape(); objs1.plot_shape(); objs2.get_shape(); objs2.plot_shape(); objs3.get_shape(); objs3.plot_shape()
        objs4 = FluorObj(typical_size=2.0); objs4.get_shape(); objs4.plot_shape()
        objs5 = FluorObj(typical_size=4.8); objs5.get_shape(); objs5.plot_shape(); objs5.get_shaping_functions()
        objs6 = FluorObj(typical_size=4.8, border_type="co", shape_method="bump3"); objs6.get_shape(); objs6.plot_shape()
    if prepare_shifted_docs_images:
        objs7 = FluorObj(typical_size=2.0, center_shifts=shifts_sample, border_type="computed", shape_method="circle")
        objs8 = FluorObj(typical_size=2.0, center_shifts=shifts_sample)
        objs7.get_shape(); objs7.plot_shape(); objs8.get_shape(); objs8.plot_shape()
        objel3 = FluorObj(shape_type='ellipse', typical_size=(4.8, 3.3, np.pi/6), center_shifts=shifts_sample)
        objel3.get_shape(); objel3.plot_shape()
    if prepare_scene_samples:
        precompile_fluoscene()  # forcing precompilation by numba
        samples1 = UscopeScene.get_random_objects(mean_size=(9.5, 8.0), size_std=(1.15, 0.82), shapes='mixed', intensity_range=(186, 254),
                                                  n_objects=25, verbose_info=True, accelerated=True)
        scene_s = UscopeScene(width=104, height=92)
        samples1_pl = scene_s.set_random_places(samples1, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene_s.put_objects_on(samples1_pl, save_only_objects_inside=True); scene_s.show_scene("Scene without noise", color_map="gray")
        if saving_possible:
            io.imsave(Path.home().joinpath("Scene_without_noise.png"), scene_s.image)
        scene_s.add_noise(); scene_s.show_scene("Scene with added noise (default parameters)", color_map="gray")
        if saving_possible:
            io.imsave(Path.home().joinpath("Scene_with_noise.png"), scene_s.image)
    # Testing acceleration by using numba compilation in the imported module
    if test_compiling_acceleration:
        precompile_fluoscene()  # force precompilation of functions by numba
        # Computing discrete round shape with the attempt to accelerate computation by using numba compilation in the module
        t_ov_1 = time.perf_counter(); objs10 = FluorObj(typical_size=12.0); objs10.get_shape(accelerated=True); objs10.plot_shape()
        elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
        if elapsed_time_ov > 1000:
            elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
            print(f"Compiled round shape computation took {elapsed_time_ov} seconds", flush=True)
        else:
            print(f"Compiled round shape computation took {elapsed_time_ov} milliseconds", flush=True)
        # Standard computing discrete round shape
        t_ov_1 = time.perf_counter(); objs11 = FluorObj(typical_size=12.0); objs11.get_shape(); objs11.plot_shape()
        elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
        if elapsed_time_ov > 1000:
            elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
            print(f"Standard round shape computation took {elapsed_time_ov} seconds", flush=True)
        else:
            print(f"Standard round shape computation took {elapsed_time_ov} milliseconds", flush=True)
        # Computing discrete ellipse shape with the attempt to accelerate computation by using numba compilation in the module
        t_ov_1 = time.perf_counter(); objs12 = FluorObj(shape_type='ellipse', typical_size=(7.5, 6.0, np.pi/3))
        objs12.get_shape(accelerated=True); objs12.plot_shape(); elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
        if elapsed_time_ov > 1000:
            elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
            print(f"Compiled ellipse shape computation took {elapsed_time_ov} seconds", flush=True)
        else:
            print(f"Compiled ellipse shape computation took {elapsed_time_ov} milliseconds", flush=True)
        # Standard computing discrete ellipse shape
        t_ov_1 = time.perf_counter(); objs13 = FluorObj(shape_type='ellipse', typical_size=(7.5, 6.0, np.pi/3))
        objs13.get_shape(); objs13.plot_shape(); elapsed_time_ov = int(round(1000.0*(time.perf_counter() - t_ov_1), 0))
        if elapsed_time_ov > 1000:
            elapsed_time_ov /= 1000.0; elapsed_time_ov = round(elapsed_time_ov, 2)
            print(f"Standard ellipse shape computation took {elapsed_time_ov} seconds", flush=True)
        else:
            print(f"Standard ellipse shape computation took {elapsed_time_ov} milliseconds", flush=True)
            del objs10, objs11, objs12, objs13
    # Prepare sample images for using as favicons in the documentation and for the main documentation page
    if prepare_favicon_img:
        round_objs2 = UscopeScene.get_round_objects(mean_size=10.0, size_std=2.0, intensity_range=(242, 252), n_objects=100)
        scene_favicon = UscopeScene(width=256, height=256)
        round_objs2 = scene_favicon.set_random_places(round_objs2, overlapping=False, touching=False, verbose_info=True)
        scene_favicon.put_objects_on(round_objs2); scene_favicon.add_noise(); scene_favicon.show_scene(color_map='gray')
        if saving_possible:
            io.imsave(Path.home().joinpath("favicon.png"), scene_favicon.image)
    if prepare_large_favicon_img:
        round_objs3 = UscopeScene.get_round_objects(mean_size=8.0, size_std=1.0, intensity_range=(240, 252), n_objects=20)
        scene_favicon2 = UscopeScene(width=64, height=64)
        round_objs3 = scene_favicon2.set_random_places(round_objs3, overlapping=False, touching=False, verbose_info=True)
        scene_favicon2.put_objects_on(round_objs3); scene_favicon2.add_noise(); scene_favicon2.show_scene(color_map='cividis')
        custom_path = ""
        if len(custom_path) > 0 and saving_possible:
            io.imsave(Path(custom_path), scene_favicon2.image)

    if test_add_noise_ext_img:
        scene4noise = UscopeScene(width=169, height=197)
        print(scene4noise.shape_types)
        objs20 = scene4noise.get_objects_acc(mean_size=(5, 10), size_std=(1, 4), intensity_range=(220, 251), shapes='mixed', n_objects=9,
                                             verbose_info=True)
        objs20 = scene4noise.set_random_places(objs20, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
        scene4noise.put_objects_on(objs20); scene4noise.show_scene()
        noisy_img = scene4noise.add_noise(mean_g=5, sigma_g=22, gain_p=0.55); scene4noise.show_scene()

    if test_cast:
        scene = UscopeScene(width=267, height=232, image_type=np.uint16)
        objs = scene.get_round_objects(mean_size=12, size_std=2, intensity_range=(15, 4090), n_objects=14, image_type=scene.img_type)
        objs = scene.set_random_places(objs); scene.put_objects_on(objs); scene.add_noise(); scene.show_scene()
        img_neg_norm = UscopeScene.cast_image(scene.image, "neg.norm."); img_int8 = UscopeScene.cast_image(scene.image, option='int8')
        img_int16 = UscopeScene.cast_image(scene.image, option='int16'); img_norm = UscopeScene.cast_image(img_int8)
        img_uint8 = UscopeScene.cast_image(img_int16, option='uint8'); img_uint16 = UscopeScene.cast_image(img_neg_norm, option='uint16')
        rng = np.random.default_rng()
        noisy_img = (rng.random(size=(156, 121)) - 0.5)*1E-3  # just noisy image shifted to (-0.5, 0.5) and scaled to small values
        if check_warning:
            norm_noisy_img = UscopeScene.cast_image(noisy_img)  # should normally generate a warning of caught noisy image
        else:
            with warnings.catch_warnings():  # context manager for collecting Warnings based on rule below
                warnings.simplefilter("ignore", UserWarning); norm_noisy_img = UscopeScene.cast_image(noisy_img)

    if show_valuable_round_objs:
        scene = UscopeScene(width=267, height=232, image_type=np.uint16)
        objs = scene.get_round_objects(mean_size=12, size_std=2, intensity_range=(15, 4090), n_objects=20, image_type=scene.img_type)
        objs = scene.set_random_places(objs); scene.put_objects_on(objs); scene.show_scene()

    if test_cleaning_compilation_cache:
        print("Local cache cleaned:", clean_fluoscene_cache())

    plt.show()

else:
    print("Check the imports, project is imported from standard installation folder 'site-packages'", flush=True)
