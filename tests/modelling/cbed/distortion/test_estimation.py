# -*- coding: utf-8 -*-
# Copyright 2025 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
r"""Contains tests for the module
:mod:`emicrocml.modelling.cbed.distortion.estimation`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For operations related to unit tests.
import pytest

# For creating distortion models.
import distoptica

# For generating fake CBED disks.
import fakecbed



# For creating wrappers to PyTorch optimizer classes.
import emicroml.modelling.optimizers

# For creating learning rate scheduler managers.
import emicroml.modelling.lr

# For training models for distortion estimation in CBED.
import emicroml.modelling.cbed.distortion.estimation



##################################
## Define classes and functions ##
##################################



def generate_default_distortion_model_generator_1_ctor_params():
    default_distortion_model_generator_1_ctor_params = \
        {"reference_pt": (0.53, 0.48),
         "rng_seed": 7,
         "sampling_grid_dims_in_pixels": (64, 64),
         "least_squares_alg_params": distoptica.LeastSquaresAlgParams(),
         "device_name": "cpu",
         "skip_validation_and_conversion": False}

    return default_distortion_model_generator_1_ctor_params



def generate_default_distortion_model_generator_2_ctor_params():
    default_distortion_model_generator_2_ctor_params = \
        {"reference_pt": (100.5, 0.5),
         "rng_seed": 366,
         "sampling_grid_dims_in_pixels": (64, 64),
         "least_squares_alg_params": None,
         "device_name": None,
         "skip_validation_and_conversion": False}

    return default_distortion_model_generator_2_ctor_params



def generate_default_cbed_pattern_generator_1_ctor_params():
    default_cbed_pattern_generator_1_ctor_params = \
        {"num_pixels_across_each_cbed_pattern": 64,
         "max_num_disks_in_any_cbed_pattern": 90,
         "rng_seed": 0,
         "sampling_grid_dims_in_pixels": (64, 64),
         "least_squares_alg_params": None,
         "device_name": None,
         "skip_validation_and_conversion": False}

    return default_cbed_pattern_generator_1_ctor_params



class InvalidCBEDPatternGenerator1():
    def __init__(self):
        return None



    def generate(self):
        return None



_module_alias = emicroml.modelling.cbed.distortion.estimation
_cls_alias = _module_alias.DefaultCBEDPatternGenerator
class InvalidCBEDPatternGenerator2(_cls_alias):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 sampling_grid_dims_in_pixels,
                 least_squares_alg_params,
                 device_name,
                 skip_validation_and_conversion):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        module_alias = emicroml.modelling.cbed.distortion.estimation
        cls_alias = module_alias.DefaultCBEDPatternGenerator
        cls_alias.__init__(self, **kwargs)

        return None



    def generate(self):
        cbed_pattern = super().generate()

        kwargs = {"center": (0.5, 0.5),
                  "radius": 2,
                  "intra_shape_val": 1}
        undistorted_disk_support = fakecbed.shapes.Circle(**kwargs)

        kwargs = {"support": undistorted_disk_support,
                  "intra_support_shapes": tuple()}
        undistorted_disk = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

        undistorted_disks = (cbed_pattern.core_attrs["undistorted_disks"]
                             + (undistorted_disk,))

        new_core_attr_subset_candidate = {"undistorted_disks": \
                                          undistorted_disks}
        cbed_pattern.update(new_core_attr_subset_candidate)

        return cbed_pattern



_module_alias = emicroml.modelling.cbed.distortion.estimation
_cls_alias = _module_alias.DefaultCBEDPatternGenerator
class InvalidCBEDPatternGenerator3(_cls_alias):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 sampling_grid_dims_in_pixels,
                 least_squares_alg_params,
                 device_name,
                 skip_validation_and_conversion):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        module_alias = emicroml.modelling.cbed.distortion.estimation
        cls_alias = module_alias.DefaultCBEDPatternGenerator
        cls_alias.__init__(self, **kwargs)

        return None



    def generate(self):
        cbed_pattern = super().generate()

        distortion_model = cbed_pattern.core_attrs["distortion_model"]

        kwargs = {"radial_cosine_coefficient_matrix": ((0.001,),)}
        coord_transform_params = distoptica.CoordTransformParams(**kwargs)

        new_core_attr_subset_candidate = {"coord_transform_params": \
                                          coord_transform_params}
        distortion_model.update(new_core_attr_subset_candidate)

        new_core_attr_subset_candidate = {"distortion_model": distortion_model}
        cbed_pattern.update(new_core_attr_subset_candidate)

        return cbed_pattern



_module_alias = emicroml.modelling.cbed.distortion.estimation
_cls_alias = _module_alias.DefaultCBEDPatternGenerator
class InvalidCBEDPatternGenerator4(_cls_alias):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 sampling_grid_dims_in_pixels,
                 least_squares_alg_params,
                 device_name,
                 skip_validation_and_conversion):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        module_alias = emicroml.modelling.cbed.distortion.estimation
        cls_alias = module_alias.DefaultCBEDPatternGenerator
        cls_alias.__init__(self, **kwargs)

        self._generation_count = 0

        return None



    def generate(self):
        cbed_pattern = super().generate()
        
        if self._generation_count > 0:
            new_core_attr_subset_candidate = \
                {"num_pixels_across_pattern": \
                 self._num_pixels_across_each_cbed_pattern//2}
            _ = \
                cbed_pattern.update(new_core_attr_subset_candidate)

        self._generation_count += 1

        return cbed_pattern



def test_1_of_DefaultDistortionModelGenerator():
    module_alias = emicroml.modelling.cbed.distortion.estimation
    cls_alias = module_alias.DefaultDistortionModelGenerator

    kwargs = generate_default_distortion_model_generator_1_ctor_params()
    distortion_model_generator = cls_alias(**kwargs)

    distortion_model_generator.validation_and_conversion_funcs
    distortion_model_generator.pre_serialization_funcs
    distortion_model_generator.de_pre_serialization_funcs

    kwargs = {"serializable_rep": distortion_model_generator.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    distortion_model_generator.generate()

    new_core_attr_subset_candidate = {"sampling_grid_dims_in_pixels": (50, 45)}
    distortion_model_generator.update(new_core_attr_subset_candidate)

    return None



def test_2_of_DefaultDistortionModelGenerator():
    module_alias = emicroml.modelling.cbed.distortion.estimation
    cls_alias = module_alias.DefaultDistortionModelGenerator

    kwargs = generate_default_distortion_model_generator_2_ctor_params()
    distortion_model_generator = cls_alias(**kwargs)

    with pytest.raises(RuntimeError) as err_info:
        distortion_model_generator.generate()

    return None



def test_1_of_DefaultCBEDPatternGenerator():
    module_alias = emicroml.modelling.cbed.distortion.estimation
    cls_alias = module_alias.DefaultCBEDPatternGenerator

    kwargs = generate_default_cbed_pattern_generator_1_ctor_params()
    cbed_pattern_generator = cls_alias(**kwargs)

    cbed_pattern_generator.validation_and_conversion_funcs
    cbed_pattern_generator.pre_serialization_funcs
    cbed_pattern_generator.de_pre_serialization_funcs

    kwargs = {"serializable_rep": cbed_pattern_generator.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    for _ in range(2):
        cbed_pattern_generator.generate()

    new_core_attr_subset_candidate = {"rng_seed": 27}
    cbed_pattern_generator.update(new_core_attr_subset_candidate)

    cbed_pattern_generator.generate()

    new_core_attr_subset_candidate = {"max_num_disks_in_any_cbed_pattern": 4,
                                      "rng_seed": 0}
    cbed_pattern_generator.update(new_core_attr_subset_candidate)

    with pytest.raises(RuntimeError) as err_info:
        cbed_pattern_generator.generate()

    return None



def test_1_of_generate_and_save_ml_dataset():
    module_alias = emicroml.modelling.cbed.distortion.estimation
    cls_alias = module_alias.DefaultCBEDPatternGenerator

    kwargs = generate_default_cbed_pattern_generator_1_ctor_params()
    valid_cbed_pattern_generator_1 = cls_alias(**kwargs)

    invalid_cbed_pattern_generator_1 = InvalidCBEDPatternGenerator1()
    invalid_cbed_pattern_generator_2 = InvalidCBEDPatternGenerator2(**kwargs)
    invalid_cbed_pattern_generator_3 = InvalidCBEDPatternGenerator3(**kwargs)
    invalid_cbed_pattern_generator_4 = InvalidCBEDPatternGenerator4(**kwargs)

    output_filename = "./test_data/modelling/cbed/distortion/ml_dataset.h5"

    kwargs = {"num_cbed_patterns": 4,
              "max_num_disks_in_any_cbed_pattern": 90,
              "cbed_pattern_generator": valid_cbed_pattern_generator_1,
              "output_filename": output_filename,
              "max_num_ml_data_instances_per_file_update": 3}
    module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(RuntimeError) as err_info:
        kwargs["cbed_pattern_generator"] = 2
        module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(TypeError) as err_info:
        kwargs["cbed_pattern_generator"] = invalid_cbed_pattern_generator_1
        module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(ValueError) as err_info:
        kwargs["cbed_pattern_generator"] = valid_cbed_pattern_generator_1
        kwargs["max_num_disks_in_any_cbed_pattern"] = 1
        module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(ValueError) as err_info:
        kwargs["cbed_pattern_generator"] = invalid_cbed_pattern_generator_2
        kwargs["max_num_disks_in_any_cbed_pattern"] = 90
        module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(ValueError) as err_info:
        kwargs["cbed_pattern_generator"] = invalid_cbed_pattern_generator_3
        module_alias.generate_and_save_ml_dataset(**kwargs)

    with pytest.raises(ValueError) as err_info:
        kwargs["cbed_pattern_generator"] = invalid_cbed_pattern_generator_4
        module_alias.generate_and_save_ml_dataset(**kwargs)

    return None



def test_1_of_combine_ml_dataset_files():
    module_alias = emicroml.modelling.cbed.distortion.estimation
    cls_alias = module_alias.DefaultCBEDPatternGenerator

    kwargs = generate_default_cbed_pattern_generator_1_ctor_params()
    valid_cbed_pattern_generator_1 = cls_alias(**kwargs)

    input_ml_dataset_filenames = tuple()
    for ml_dataset_idx in range(2):
        output_filename = ("./test_data/modelling/cbed/distortion"
                           "/ml_dataset_{}.h5").format(ml_dataset_idx)
        input_ml_dataset_filenames += (output_filename,)

        kwargs = {"num_cbed_patterns": 2,
                  "max_num_disks_in_any_cbed_pattern": 90,
                  "cbed_pattern_generator": valid_cbed_pattern_generator_1,
                  "output_filename": output_filename,
                  "max_num_ml_data_instances_per_file_update": 3}
        module_alias.generate_and_save_ml_dataset(**kwargs)

    output_ml_dataset_filename = ("./test_data/modelling/cbed/distortion"
                                  "/ml_dataset.h5")

    for rm_input_ml_dataset_files in (False, True):
        kwargs = {"input_ml_dataset_filenames": input_ml_dataset_filenames,
                  "output_ml_dataset_filename": output_ml_dataset_filename,
                  "rm_input_ml_dataset_files": rm_input_ml_dataset_files,
                  "max_num_ml_data_instances_per_file_update": 3}
        module_alias.combine_ml_dataset_files(**kwargs)

    # _apply_corruption_scheme_1_to_ml_dataset_file(ml_dataset_filename)

    return None



###########################
## Define error messages ##
###########################
