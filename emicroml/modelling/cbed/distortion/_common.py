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
"""Contains code common to various modules in the subpackage
:mod:`emicroml.modelling.cbed.distortion`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For generating the alphabet.
import string



# For general array handling.
import numpy as np

# For generating distortion models.
import distoptica

# For generating fake CBED patterns.
import fakecbed

# For building neural network models.
import torch

# For image processing tools that can be integrated into deep learning models.
import kornia

# For validating and converting objects.
import czekitout.convert

# For closing HDF5 files.
import h5py



# Contains implementation code that is applicable to the current module.
import emicroml.modelling._common
import emicroml.modelling.cbed._common



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = []



def _get_device(device_name):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_get_device"
    func_alias = getattr(module_alias, func_name)
    device = func_alias(**kwargs)

    return device



def _check_and_convert_reference_pt(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_reference_pt"
    func_alias = getattr(module_alias, func_name)
    reference_pt = func_alias(**kwargs)

    return reference_pt



def _pre_serialize_reference_pt(reference_pt):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_reference_pt"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_reference_pt(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_reference_pt"
    func_alias = getattr(module_alias, func_name)
    reference_pt = func_alias(**kwargs)

    return reference_pt



def _check_and_convert_rng_seed(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_rng_seed"
    func_alias = getattr(module_alias, func_name)
    rng_seed = func_alias(**kwargs)

    return rng_seed



def _pre_serialize_rng_seed(rng_seed):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_rng_seed"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_rng_seed(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_rng_seed"
    func_alias = getattr(module_alias, func_name)
    rng_seed = func_alias(**kwargs)

    return rng_seed



def _check_and_convert_sampling_grid_dims_in_pixels(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_sampling_grid_dims_in_pixels"
    func_alias = getattr(module_alias, func_name)
    sampling_grid_dims_in_pixels = func_alias(**kwargs)

    return sampling_grid_dims_in_pixels



def _pre_serialize_sampling_grid_dims_in_pixels(sampling_grid_dims_in_pixels):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_sampling_grid_dims_in_pixels"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_sampling_grid_dims_in_pixels(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_sampling_grid_dims_in_pixels"
    func_alias = getattr(module_alias, func_name)
    sampling_grid_dims_in_pixels = func_alias(**kwargs)

    return sampling_grid_dims_in_pixels



def _check_and_convert_least_squares_alg_params(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_least_squares_alg_params"
    func_alias = getattr(module_alias, func_name)
    least_squares_alg_params = func_alias(**kwargs)

    return least_squares_alg_params



def _pre_serialize_least_squares_alg_params(least_squares_alg_params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_least_squares_alg_params"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_least_squares_alg_params(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_least_squares_alg_params"
    func_alias = getattr(module_alias, func_name)
    least_squares_alg_params = func_alias(**kwargs)

    return least_squares_alg_params



def _check_and_convert_device_name(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_device_name"
    func_alias = getattr(module_alias, func_name)
    device_name = func_alias(**kwargs)

    return device_name



def _pre_serialize_device_name(device_name):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_device_name"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_device_name(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_device_name"
    func_alias = getattr(module_alias, func_name)
    device_name = func_alias(**kwargs)

    return device_name



_module_alias = \
    emicroml.modelling.cbed._common
_default_reference_pt = \
    _module_alias._default_reference_pt
_default_rng_seed = \
    _module_alias._default_rng_seed
_default_sampling_grid_dims_in_pixels = \
    _module_alias._default_sampling_grid_dims_in_pixels
_default_least_squares_alg_params = \
    _module_alias._default_least_squares_alg_params
_default_device_name = \
    _module_alias._default_device_name
_default_skip_validation_and_conversion = \
    _module_alias._default_skip_validation_and_conversion



_module_alias = emicroml.modelling.cbed._common
_cls_alias = _module_alias._DefaultDistortionModelGenerator
class _DefaultDistortionModelGenerator(_cls_alias):
    def __init__(self,
                 reference_pt,
                 rng_seed,
                 sampling_grid_dims_in_pixels,
                 least_squares_alg_params,
                 device_name,
                 skip_validation_and_conversion):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        module_alias = emicroml.modelling.cbed._common
        cls_alias = module_alias._DefaultDistortionModelGenerator
        cls_alias.__init__(self, **kwargs)

        return None



_building_block_counts_in_stages_of_distoptica_net = \
    (3, 5, 2)



def _check_and_convert_num_pixels_across_each_cbed_pattern(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_num_pixels_across_each_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    num_pixels_across_each_cbed_pattern = func_alias(**kwargs)

    max_num_downsampling_steps_in_any_encoder_used_in_ml_model = \
        (emicroml.modelling._common._DistopticaNetEntryFlow._num_downsamplings
         + len(_building_block_counts_in_stages_of_distoptica_net))

    current_func_name = "_check_and_convert_num_pixels_across_each_cbed_pattern"

    M = 2**max_num_downsampling_steps_in_any_encoder_used_in_ml_model
    if num_pixels_across_each_cbed_pattern % M != 0:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(M)
        raise ValueError(err_msg)

    return num_pixels_across_each_cbed_pattern



def _pre_serialize_num_pixels_across_each_cbed_pattern(
        num_pixels_across_each_cbed_pattern):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_num_pixels_across_each_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_num_pixels_across_each_cbed_pattern(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_num_pixels_across_each_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    num_pixels_across_each_cbed_pattern = func_alias(**kwargs)

    return num_pixels_across_each_cbed_pattern



def _check_and_convert_max_num_disks_in_any_cbed_pattern(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_max_num_disks_in_any_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    max_num_disks_in_any_cbed_pattern = func_alias(**kwargs)

    return max_num_disks_in_any_cbed_pattern



def _pre_serialize_max_num_disks_in_any_cbed_pattern(
        max_num_disks_in_any_cbed_pattern):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_pre_serialize_max_num_disks_in_any_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    serializable_rep = func_alias(**kwargs)
    
    return serializable_rep



def _de_pre_serialize_max_num_disks_in_any_cbed_pattern(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_max_num_disks_in_any_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    max_num_disks_in_any_cbed_pattern = func_alias(**kwargs)

    return max_num_disks_in_any_cbed_pattern



_module_alias = \
    emicroml.modelling.cbed._common
_default_num_pixels_across_each_cbed_pattern = \
    _module_alias._default_num_pixels_across_each_cbed_pattern
_default_max_num_disks_in_any_cbed_pattern = \
    _module_alias._default_max_num_disks_in_any_cbed_pattern



_module_alias = emicroml.modelling.cbed._common
_cls_alias = _module_alias._DefaultCBEDPatternGenerator
class _DefaultCBEDPatternGenerator(_cls_alias):
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
        module_alias = emicroml.modelling.cbed._common
        cls_alias = module_alias._DefaultCBEDPatternGenerator
        cls_alias.__init__(self, **kwargs)

        return None



def _generate_keys_of_unnormalizable_ml_data_dict_elems():
    unformatted_func_name = ("_generate_keys_of_unnormalizable"
                             "_ml_data_dict_elems{}_having_decoders")

    keys_of_unnormalizable_ml_data_dict_elems = tuple()
    global_symbol_table = globals()
    for format_arg in ("_not", ""):
        args = (format_arg,)
        func_name = unformatted_func_name.format(*args)
        func_alias = global_symbol_table[func_name]
        keys_of_unnormalizable_ml_data_dict_elems += func_alias()

    return keys_of_unnormalizable_ml_data_dict_elems



def _generate_keys_of_unnormalizable_ml_data_dict_elems_not_having_decoders():
    keys_of_unnormalizable_ml_data_dict_elems_not_having_decoders = \
        ("cbed_pattern_images",
         "disk_overlap_maps",
         "disk_clipping_registries",
         "disk_objectness_sets")

    return keys_of_unnormalizable_ml_data_dict_elems_not_having_decoders



def _generate_keys_of_unnormalizable_ml_data_dict_elems_having_decoders():
    module_alias = emicroml.modelling.cbed._common
    func_name = ("_generate_keys_of_unnormalizable"
                 "_ml_data_dict_elems_having_decoders")
    func_alias = getattr(module_alias, func_name)
    keys_of_unnormalizable_ml_data_dict_elems_having_decoders = func_alias()

    return keys_of_unnormalizable_ml_data_dict_elems_having_decoders



def _generate_keys_of_normalizable_ml_data_dict_elems():
    keys_of_normalizable_ml_data_dict_elems = \
        _generate_keys_of_normalizable_ml_data_dict_elems_not_having_decoders()
    keys_of_normalizable_ml_data_dict_elems += \
         _generate_keys_of_normalizable_ml_data_dict_elems_having_decoders()

    return keys_of_normalizable_ml_data_dict_elems



def _generate_keys_of_normalizable_ml_data_dict_elems_not_having_decoders():
    keys_of_normalizable_ml_data_dict_elems_not_having_decoders = \
        ("common_undistorted_disk_radii", "undistorted_disk_center_sets")
    keys_of_normalizable_ml_data_dict_elems_not_having_decoders += \
        _generate_keys_related_to_distortion_params()

    return keys_of_normalizable_ml_data_dict_elems_not_having_decoders



def _generate_keys_related_to_distortion_params():
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_keys_related_to_distortion_params"
    func_alias = getattr(module_alias, func_name)
    keys_related_to_distortion_params = func_alias()

    return keys_related_to_distortion_params



def _generate_keys_of_normalizable_ml_data_dict_elems_having_decoders():
    module_alias = emicroml.modelling.cbed._common
    func_name = ("_generate_keys_of_normalizable"
                 "_ml_data_dict_elems_having_decoders")
    func_alias = getattr(module_alias, func_name)
    keys_of_normalizable_ml_data_dict_elems_having_decoders = func_alias()

    return keys_of_normalizable_ml_data_dict_elems_having_decoders



def _generate_all_valid_ml_data_dict_keys():
    ml_data_dict_keys = _generate_keys_of_unnormalizable_ml_data_dict_elems()
    ml_data_dict_keys += _generate_keys_of_normalizable_ml_data_dict_elems()
    
    return ml_data_dict_keys



def _generate_cbed_pattern_signal(cbed_pattern_generator):
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_cbed_pattern_signal"
    func_alias = getattr(module_alias, func_name)
    kwargs = {"cbed_pattern_generator": cbed_pattern_generator}
    cbed_pattern_signal = func_alias(**kwargs)

    return cbed_pattern_signal



def _check_cbed_pattern_signal(cbed_pattern_signal,
                               max_num_disks_in_any_cbed_pattern):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_cbed_pattern_signal"
    func_alias = getattr(module_alias, func_name)
    func_alias(**kwargs)

    return None



def _extract_ml_data_dict_from_cbed_pattern_signal(
        cbed_pattern_signal, max_num_disks_in_any_cbed_pattern):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_extract_ml_data_dict_from_cbed_pattern_signal"
    func_alias = getattr(module_alias, func_name)
    ml_data_dict = func_alias(**kwargs)

    return ml_data_dict



_module_alias = emicroml.modelling.cbed._common
_tol_for_comparing_floats = _module_alias._tol_for_comparing_floats



def _de_pre_serialize_coord_transform_params(serializable_rep):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_de_pre_serialize_coord_transform_params"
    func_alias = getattr(module_alias, func_name)
    coord_transform_params = func_alias(**kwargs)

    return coord_transform_params
        



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._UnnormalizedMLDataInstanceGenerator
class _UnnormalizedMLDataInstanceGenerator(_cls_alias):
    def __init__(self,
                 cbed_pattern_generator,
                 max_num_disks_in_any_cbed_pattern):
        self._cbed_pattern_generator = \
            cbed_pattern_generator
        self._max_num_disks_in_any_cbed_pattern = \
            max_num_disks_in_any_cbed_pattern

        self._expected_cbed_pattern_dims_in_pixels = None
        
        cbed_pattern_signal = \
            _generate_cbed_pattern_signal(cbed_pattern_generator)
        self._expected_cbed_pattern_dims_in_pixels = \
            cbed_pattern_signal.axes_manager.signal_shape

        cached_ml_data_instances = self._generate(num_ml_data_instances=1)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._UnnormalizedMLDataInstanceGenerator
        kwargs = {"cached_ml_data_instances": cached_ml_data_instances}
        cls_alias.__init__(self, **kwargs)

        return None



    def _generate_ml_data_dict_containing_only_one_ml_data_instance(self):
        max_num_disks_in_any_cbed_pattern = \
            self._max_num_disks_in_any_cbed_pattern
        cbed_pattern_generator = \
            self._cbed_pattern_generator
        cbed_pattern_signal = \
            _generate_cbed_pattern_signal(cbed_pattern_generator)
        cbed_pattern_dims_in_pixels = \
            cbed_pattern_signal.axes_manager.signal_shape
        expected_cbed_pattern_dims_in_pixels = \
            self._expected_cbed_pattern_dims_in_pixels

        _check_cbed_pattern_signal(cbed_pattern_signal,
                                   max_num_disks_in_any_cbed_pattern)

        if cbed_pattern_dims_in_pixels != expected_cbed_pattern_dims_in_pixels:
            err_msg = _unnormalized_ml_data_instance_generator_err_msg_1
            raise ValueError(err_msg)

        self._expected_cbed_pattern_dims_in_pixels = \
            cbed_pattern_signal.axes_manager.signal_shape

        method_alias = \
            super()._generate_ml_data_dict_containing_only_one_ml_data_instance
        ml_data_dict = \
            method_alias()

        func_alias = _extract_ml_data_dict_from_cbed_pattern_signal
        kwargs = {"cbed_pattern_signal": \
                  cbed_pattern_signal,
                  "max_num_disks_in_any_cbed_pattern": \
                  max_num_disks_in_any_cbed_pattern}
        ml_data_dict = {**ml_data_dict, **func_alias(**kwargs)}

        return ml_data_dict



def _generate_ml_data_dict_elem_decoders():
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_ml_data_dict_elem_decoders"
    func_alias = getattr(module_alias, func_name)
    ml_data_dict_elem_decoders = func_alias(**kwargs)

    return ml_data_dict_elem_decoders



def _generate_ml_data_dict_key_to_shape_template_map():
    ml_data_dict_key_to_shape_template_map = dict()

    variable_axis_size_dict_keys = _generate_variable_axis_size_dict_keys()
    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key in all_valid_ml_data_dict_keys:
        if key in ("cbed_pattern_images", "disk_overlap_maps"):
            shape_template = (variable_axis_size_dict_keys[0],
                              variable_axis_size_dict_keys[1],
                              variable_axis_size_dict_keys[1])
        elif key in ("disk_objectness_sets", "disk_clipping_registries"):
            shape_template = (variable_axis_size_dict_keys[0],
                              variable_axis_size_dict_keys[2])
        elif key == "undistorted_disk_center_sets":
            shape_template = (variable_axis_size_dict_keys[0],
                              variable_axis_size_dict_keys[2],
                              2)
        elif ("centers" in key) or ("vectors" in key):
            shape_template = (variable_axis_size_dict_keys[0], 2)
        else:
            shape_template = (variable_axis_size_dict_keys[0],)

        ml_data_dict_key_to_shape_template_map[key] = shape_template

    return ml_data_dict_key_to_shape_template_map



def _generate_variable_axis_size_dict_keys():
    num_keys = 3
    
    alphabet = tuple(string.ascii_uppercase)
    subset_of_alphabet = alphabet[:num_keys]

    variable_axis_size_dict_keys = tuple("axis size "+letter
                                         for letter
                                         in subset_of_alphabet)

    return variable_axis_size_dict_keys



def _generate_ml_data_dict_elem_decoding_order():
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_ml_data_dict_elem_decoding_order"
    func_alias = getattr(module_alias, func_name)
    ml_data_dict_elem_decoding_order = func_alias(**kwargs)

    return ml_data_dict_elem_decoding_order



def _generate_overriding_normalization_weights_and_biases():
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_overriding_normalization_weights_and_biases"
    func_alias = getattr(module_alias, func_name)
    overriding_normalization_weights_and_biases = func_alias(**kwargs)

    return overriding_normalization_weights_and_biases



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataNormalizer
class _MLDataNormalizer(_cls_alias):
    def __init__(self, max_num_ml_data_instances_per_file_update):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataNormalizer
        kwargs = {"keys_of_unnormalizable_ml_data_dict_elems": \
                  _generate_keys_of_unnormalizable_ml_data_dict_elems(),
                  "keys_of_normalizable_ml_data_dict_elems": \
                  _generate_keys_of_normalizable_ml_data_dict_elems(),
                  "ml_data_dict_elem_decoders": \
                  _generate_ml_data_dict_elem_decoders(),
                  "overriding_normalization_weights_and_biases": \
                  _generate_overriding_normalization_weights_and_biases(),
                  "max_num_ml_data_instances_per_file_update": \
                  max_num_ml_data_instances_per_file_update}
        cls_alias.__init__(self, **kwargs)

        return None



_module_alias = \
    emicroml.modelling.cbed._common
_default_max_num_ml_data_instances_per_file_update = \
    _module_alias._default_max_num_ml_data_instances_per_file_update



def _generate_default_ml_data_normalizer():
    kwargs = {"max_num_ml_data_instances_per_file_update": \
              _default_max_num_ml_data_instances_per_file_update}
    ml_data_normalizer = _MLDataNormalizer(**kwargs)

    return ml_data_normalizer



def _generate_ml_data_dict_key_to_dtype_map():
    ml_data_dict_key_to_dtype_map = dict()

    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key in all_valid_ml_data_dict_keys:
        if "clipping" in key:
            dtype = np.bool_
        elif "overlap_map" in key:
            dtype = np.uint8
        else:
            dtype = np.float32
            
        ml_data_dict_key_to_dtype_map[key] = dtype

    return ml_data_dict_key_to_dtype_map



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataTypeValidator
class _MLDataTypeValidator(_cls_alias):
    def __init__(self):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataTypeValidator
        kwargs = {"ml_data_dict_key_to_dtype_map": \
                  _generate_ml_data_dict_key_to_dtype_map()}
        cls_alias.__init__(self, **kwargs)

        return None



def _generate_ml_data_dict_key_to_unnormalized_value_limits_map():
    ml_data_dict_key_to_unnormalized_value_limits_map = dict()

    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key in all_valid_ml_data_dict_keys:
        if ((key == "cbed_pattern_images")
            or (key == "disk_clipping_registries")
            or (key == "common_undistorted_disk_radii")
            or (key == "disk_objectness_sets")):
            unnormalized_value_limits = (0, 1)
        elif key == "disk_overlap_maps":
            unnormalized_value_limits = (0, np.inf)
        else:
            unnormalized_value_limits = (-np.inf, np.inf)
                
        ml_data_dict_key_to_unnormalized_value_limits_map[key] = \
            unnormalized_value_limits

    return ml_data_dict_key_to_unnormalized_value_limits_map



def _generate_ml_data_dict_key_to_custom_value_checker_map():
    ml_data_dict_key_to_custom_value_checker_map = \
        {"cbed_pattern_images": _custom_value_checker_for_cbed_pattern_images}

    return ml_data_dict_key_to_custom_value_checker_map



def _custom_value_checker_for_cbed_pattern_images(
        data_chunk_is_expected_to_be_normalized_if_normalizable,
        key_used_to_get_data_chunk,
        data_chunk,
        name_of_obj_alias_from_which_data_chunk_was_obtained,
        obj_alias_from_which_data_chunk_was_obtained):
    lower_value_limit = 0
    upper_value_limit = 1
    tol = _tol_for_comparing_floats
    current_func_name = "_custom_value_checker_for_cbed_pattern_images"

    for cbed_pattern_image in data_chunk:
        if ((abs(cbed_pattern_image.min().item()-lower_value_limit) > tol)
            or (abs(cbed_pattern_image.max().item()-upper_value_limit) > tol)):
            obj_alias = obj_alias_from_which_data_chunk_was_obtained

            unformatted_err_msg = \
                (globals()[current_func_name+"_err_msg_1"]
                 if isinstance(obj_alias, h5py.Dataset)
                 else globals()[current_func_name+"_err_msg_2"])

            format_arg_1 = \
                (key_used_to_get_data_chunk
                 if isinstance(obj_alias, h5py.Dataset)
                 else name_of_obj_alias_from_which_data_chunk_was_obtained)
            format_arg_2 = \
                (obj_alias.file.filename
                 if isinstance(obj_alias, h5py.Dataset)
                 else key_used_to_get_data_chunk)

            args = (format_arg_1, format_arg_2)
            
            err_msg = unformatted_err_msg.format(*args)
            raise ValueError(err_msg)

    return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataValueValidator
class _MLDataValueValidator(_cls_alias):
    def __init__(self):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataValueValidator
        kwargs = {"ml_data_dict_key_to_unnormalized_value_limits_map": \
                  _generate_ml_data_dict_key_to_unnormalized_value_limits_map(),
                  "ml_data_dict_key_to_custom_value_checker_map": \
                  _generate_ml_data_dict_key_to_custom_value_checker_map(),
                  "ml_data_normalizer": \
                  _generate_default_ml_data_normalizer()}
        cls_alias.__init__(self, **kwargs)

        return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataNormalizationWeightsAndBiasesLoader
class _MLDataNormalizationWeightsAndBiasesLoader(_cls_alias):
    def __init__(self, max_num_ml_data_instances_per_file_update):
        ml_data_normalizer = \
            _MLDataNormalizer(max_num_ml_data_instances_per_file_update)
        ml_data_value_validator = \
            _MLDataValueValidator()

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataNormalizationWeightsAndBiasesLoader
        kwargs = {"ml_data_normalizer": ml_data_normalizer,
                  "ml_data_value_validator": ml_data_value_validator}
        cls_alias.__init__(self, **kwargs)
        
        return None



def _generate_default_ml_data_normalization_weights_and_biases_loader():
    kwargs = \
        {"max_num_ml_data_instances_per_file_update": 1}
    ml_data_normalization_weights_and_biases_loader = \
        _MLDataNormalizationWeightsAndBiasesLoader(**kwargs)

    return ml_data_normalization_weights_and_biases_loader



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataRenormalizer
class _MLDataRenormalizer(_cls_alias):
    def __init__(self,
                 input_ml_dataset_filenames,
                 max_num_ml_data_instances_per_file_update):
        kwargs = \
            {"max_num_ml_data_instances_per_file_update": \
             max_num_ml_data_instances_per_file_update}
        ml_data_normalization_weights_and_biases_loader = \
            _MLDataNormalizationWeightsAndBiasesLoader(**kwargs)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataRenormalizer
        kwargs = {"input_ml_dataset_filenames": \
                  input_ml_dataset_filenames,
                  "max_num_ml_data_instances_per_file_update": \
                  max_num_ml_data_instances_per_file_update,
                  "ml_data_normalization_weights_and_biases_loader": \
                  ml_data_normalization_weights_and_biases_loader}
        cls_alias.__init__(self, **kwargs)

        return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataShapeAnalyzer
class _MLDataShapeAnalyzer(_cls_alias):
    def __init__(self):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataShapeAnalyzer
        kwargs = {"variable_axis_size_dict_keys": \
                  _generate_variable_axis_size_dict_keys(),
                  "ml_data_dict_key_to_shape_template_map": \
                  _generate_ml_data_dict_key_to_shape_template_map(),
                  "ml_data_dict_elem_decoders": \
                  _generate_ml_data_dict_elem_decoders()}
        cls_alias.__init__(self, **kwargs)

        return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataSplitter
class _MLDataSplitter(_cls_alias):
    def __init__(self,
                 input_ml_dataset_filename,
                 enable_shuffling,
                 rng_seed,
                 max_num_ml_data_instances_per_file_update,
                 split_ratio):
        kwargs = \
            {"max_num_ml_data_instances_per_file_update": \
             max_num_ml_data_instances_per_file_update}
        ml_data_normalization_weights_and_biases_loader = \
            _MLDataNormalizationWeightsAndBiasesLoader(**kwargs)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataSplitter
        kwargs = {"input_ml_dataset_filename": \
                  input_ml_dataset_filename,
                  "ml_data_normalization_weights_and_biases_loader": \
                  ml_data_normalization_weights_and_biases_loader,
                  "enable_shuffling": \
                  enable_shuffling,
                  "rng_seed": \
                  rng_seed,
                  "max_num_ml_data_instances_per_file_update": \
                  max_num_ml_data_instances_per_file_update,
                  "split_ratio": \
                  split_ratio}
        cls_alias.__init__(self, **kwargs)

        return None
    


def _generate_axes_labels_of_hdf5_datasets_of_ml_dataset_file():
    axes_labels_of_hdf5_datasets_of_ml_dataset_file = dict()
    
    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key in all_valid_ml_data_dict_keys:
        if key == "cbed_pattern_images":
            axes_labels_of_hdf5_dataset = ("cbed pattern idx", "row", "col")
        elif key == "disk_overlap_maps":
            axes_labels_of_hdf5_dataset = ("cbed pattern idx", "row", "col")
        elif key in ("disk_objectness_sets", "disk_clipping_registries"):
            axes_labels_of_hdf5_dataset = ("cbed pattern idx", "disk idx")
        elif key == "undistorted_disk_center_sets":
            axes_labels_of_hdf5_dataset = ("cbed pattern idx",
                                           "disk idx",
                                           "vector cmpnt idx [0->x, 1->y]")
        elif ("centers" in key) or ("vectors" in key):
            axes_labels_of_hdf5_dataset = ("cbed pattern idx",
                                           "vector cmpnt idx [0->x, 1->y]")
        else:
            axes_labels_of_hdf5_dataset = ("cbed pattern idx",)
        
        axes_labels_of_hdf5_datasets_of_ml_dataset_file[key] = \
            axes_labels_of_hdf5_dataset

    return axes_labels_of_hdf5_datasets_of_ml_dataset_file



def _check_and_convert_generate_and_save_ml_dataset_params(params):
    params = params.copy()

    module_alias = \
        emicroml.modelling._common
    func_alias = \
        module_alias._check_and_convert_generate_and_save_ml_dataset_params
    params = \
        func_alias(params)

    param_name_subset = ("num_cbed_patterns",
                         "cbed_pattern_generator",
                         "max_num_disks_in_any_cbed_pattern")

    global_symbol_table = globals()
    for param_name in param_name_subset:
        func_name = "_check_and_convert_" + param_name
        func_alias = global_symbol_table[func_name]
        params[param_name] = func_alias(params)

    return params



def _check_and_convert_num_cbed_patterns(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_num_cbed_patterns"
    func_alias = getattr(module_alias, func_name)
    num_cbed_patterns = func_alias(**kwargs)

    return num_cbed_patterns



def _check_and_convert_cbed_pattern_generator(params):
    params["default_cbed_pattern_generator_cls"] = _DefaultCBEDPatternGenerator
    
    kwargs = {"params": params}
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_cbed_pattern_generator"
    func_alias = getattr(module_alias, func_name)
    cbed_pattern_generator = func_alias(**kwargs)

    del params["default_cbed_pattern_generator_cls"]

    return cbed_pattern_generator



_module_alias = emicroml.modelling.cbed._common
_default_cbed_pattern_generator = _module_alias._default_cbed_pattern_generator
_default_num_cbed_patterns = _module_alias._default_num_cbed_patterns
_default_output_filename = _module_alias._default_output_filename



def _generate_and_save_ml_dataset(cbed_pattern_generator,
                                  max_num_disks_in_any_cbed_pattern,
                                  max_num_ml_data_instances_per_file_update,
                                  num_cbed_patterns,
                                  output_filename,
                                  start_time):
    kwargs = \
        {"cbed_pattern_generator":
         cbed_pattern_generator,
         "max_num_disks_in_any_cbed_pattern": \
         max_num_disks_in_any_cbed_pattern}
    unnormalized_ml_data_instance_generator = \
        _UnnormalizedMLDataInstanceGenerator(**kwargs)

    kwargs = {"max_num_ml_data_instances_per_file_update": \
              max_num_ml_data_instances_per_file_update}
    ml_data_normalizer = _MLDataNormalizer(**kwargs)

    num_ml_data_instances = num_cbed_patterns

    ml_data_type_validator = \
        _MLDataTypeValidator()
    ml_data_dict_key_to_dtype_map = \
        ml_data_type_validator._ml_data_dict_key_to_dtype_map

    func_alias = _generate_axes_labels_of_hdf5_datasets_of_ml_dataset_file
    axes_labels_of_hdf5_datasets_of_ml_dataset_file = func_alias()

    module_alias = emicroml.modelling._common
    func_alias = module_alias._generate_and_save_ml_dataset
    func_alias(output_filename,
               unnormalized_ml_data_instance_generator,
               ml_data_normalizer,
               num_ml_data_instances,
               ml_data_dict_key_to_dtype_map,
               axes_labels_of_hdf5_datasets_of_ml_dataset_file,
               start_time)

    return None



def _check_and_convert_combine_ml_dataset_files_params(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_combine_ml_dataset_files_params"
    func_alias = getattr(module_alias, func_name)
    params = func_alias(**kwargs)

    return params



_module_alias = \
    emicroml.modelling.cbed._common
_default_output_ml_dataset_filename = \
    _module_alias._default_output_ml_dataset_filename
_default_rm_input_ml_dataset_files = \
    _module_alias._default_rm_input_ml_dataset_files



def _combine_ml_dataset_files(max_num_ml_data_instances_per_file_update,
                              input_ml_dataset_filenames,
                              output_ml_dataset_filename,
                              rm_input_ml_dataset_files,
                              start_time):
    ml_data_type_validator = _MLDataTypeValidator()

    ml_data_shape_analyzer = _MLDataShapeAnalyzer()

    func_alias = _generate_axes_labels_of_hdf5_datasets_of_ml_dataset_file
    axes_labels_of_hdf5_datasets_of_ml_dataset_file = func_alias()

    kwargs = {"input_ml_dataset_filenames": \
              input_ml_dataset_filenames,
              "max_num_ml_data_instances_per_file_update": \
              max_num_ml_data_instances_per_file_update}
    ml_data_renormalizer = _MLDataRenormalizer(**kwargs)

    module_alias = emicroml.modelling._common
    func_alias = module_alias._combine_ml_dataset_files
    func_alias(input_ml_dataset_filenames,
               output_ml_dataset_filename,
               ml_data_type_validator,
               ml_data_shape_analyzer,
               axes_labels_of_hdf5_datasets_of_ml_dataset_file,
               ml_data_renormalizer,
               rm_input_ml_dataset_files,
               start_time)

    return None



def _check_and_convert_split_ml_dataset_file_params(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_split_ml_dataset_file_params"
    func_alias = getattr(module_alias, func_name)
    params = func_alias(**kwargs)

    return params



_module_alias = \
    emicroml.modelling.cbed._common
_default_output_ml_dataset_filename_1 = \
    _module_alias._default_output_ml_dataset_filename_1
_default_output_ml_dataset_filename_2 = \
    _module_alias._default_output_ml_dataset_filename_2
_default_output_ml_dataset_filename_3 = \
    _module_alias._default_output_ml_dataset_filename_3
_default_split_ratio = \
    _module_alias._default_split_ratio
_default_enable_shuffling = \
    _module_alias._default_enable_shuffling
_default_rm_input_ml_dataset_file = \
    _module_alias._default_rm_input_ml_dataset_file



def _split_ml_dataset_file(output_ml_dataset_filename_1,
                           output_ml_dataset_filename_2,
                           output_ml_dataset_filename_3,
                           max_num_ml_data_instances_per_file_update,
                           input_ml_dataset_filename,
                           split_ratio,
                           enable_shuffling,
                           rng_seed,
                           rm_input_ml_dataset_file,
                           start_time):
    output_ml_dataset_filenames = (output_ml_dataset_filename_1,
                                   output_ml_dataset_filename_2,
                                   output_ml_dataset_filename_3)

    ml_data_type_validator = _MLDataTypeValidator()

    ml_data_shape_analyzer = _MLDataShapeAnalyzer()

    func_alias = _generate_axes_labels_of_hdf5_datasets_of_ml_dataset_file
    axes_labels_of_hdf5_datasets_of_ml_dataset_file = func_alias()

    kwargs = {"input_ml_dataset_filename": \
              input_ml_dataset_filename,
              "enable_shuffling": \
              enable_shuffling,
              "rng_seed": \
              rng_seed,
              "max_num_ml_data_instances_per_file_update": \
              max_num_ml_data_instances_per_file_update,
              "split_ratio": split_ratio}
    ml_data_splitter = _MLDataSplitter(**kwargs)

    module_alias = emicroml.modelling._common
    func_alias = module_alias._split_ml_dataset_file
    func_alias(ml_data_splitter,
               output_ml_dataset_filenames,
               ml_data_type_validator,
               ml_data_shape_analyzer,
               axes_labels_of_hdf5_datasets_of_ml_dataset_file,
               rm_input_ml_dataset_file,
               start_time)

    return None



def _get_num_pixels_across_each_cbed_pattern(path_to_ml_dataset,
                                             ml_data_shape_analyzer):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_get_num_pixels_across_each_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    num_pixels_across_each_cbed_pattern = func_alias(**kwargs)

    return num_pixels_across_each_cbed_pattern



def _get_max_num_disks_in_any_cbed_pattern(path_to_ml_dataset,
                                           ml_data_shape_analyzer):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_get_max_num_disks_in_any_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    max_num_disks_in_any_cbed_pattern = func_alias(**kwargs)

    return max_num_disks_in_any_cbed_pattern



def _check_and_convert_normalize_normalizable_elems_in_ml_data_dict_params(
        params):
    original_params = params
    params = params.copy()

    params["ml_data_shape_analyzer"] = \
        _MLDataShapeAnalyzer()
    params["ml_data_type_validator"] = \
        _MLDataTypeValidator()
    params["ml_data_normalization_weights_and_biases_loader"] = \
        _generate_default_ml_data_normalization_weights_and_biases_loader()
    params["default_normalization_weights"] = \
        _generate_default_normalization_weights()
    params["default_normalization_biases"] = \
        _generate_default_normalization_biases()

    current_func_name = ("_check_and_convert"
                         "_normalize_normalizable_elems_in_ml_data_dict_params")

    module_alias = emicroml.modelling._common
    func_alias = getattr(module_alias, current_func_name)
    params = func_alias(params)

    for key in tuple(params.keys()):
        if key not in original_params:
            del params[key]

    return params



_module_alias = \
    emicroml.modelling.cbed._common
_default_check_ml_data_dict_first = \
    _module_alias._default_check_ml_data_dict_first



def _normalize_normalizable_elems_in_ml_data_dict(ml_data_dict,
                                                  normalization_weights,
                                                  normalization_biases):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_normalize_normalizable_elems_in_ml_data_dict"
    func_alias = getattr(module_alias, func_name)
    func_alias(**kwargs)

    return None



def _check_and_convert_unnormalize_normalizable_elems_in_ml_data_dict_params(
        params):
    original_params = params
    params = params.copy()

    params["ml_data_shape_analyzer"] = \
        _MLDataShapeAnalyzer()
    params["ml_data_type_validator"] = \
        _MLDataTypeValidator()
    params["ml_data_normalization_weights_and_biases_loader"] = \
        _generate_default_ml_data_normalization_weights_and_biases_loader()
    params["default_normalization_weights"] = \
        _generate_default_normalization_weights()
    params["default_normalization_biases"] = \
        _generate_default_normalization_biases()

    current_func_name = ("_check_and_convert_unnormalize_normalizable_elems"
                         "_in_ml_data_dict_params")
    
    module_alias = emicroml.modelling._common
    func_alias = getattr(module_alias, current_func_name)
    params = func_alias(params)

    for key in tuple(params.keys()):
        if key not in original_params:
            del params[key]

    return params



def _unnormalize_normalizable_elems_in_ml_data_dict(ml_data_dict,
                                                    normalization_weights,
                                                    normalization_biases):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_unnormalize_normalizable_elems_in_ml_data_dict"
    func_alias = getattr(module_alias, func_name)
    func_alias(**kwargs)

    return None



def _check_and_convert_ml_data_dict_to_distortion_models_params(params):
    original_params = params
    params = params.copy()
    
    params["data_chunk_dims_are_to_be_expanded_temporarily"] = \
        False
    params["expected_ml_data_dict_keys"] = \
        _generate_keys_related_to_distortion_params()
    params["ml_data_dict"] = \
        _check_and_convert_ml_data_dict(params)
    params["sampling_grid_dims_in_pixels"] = \
        _check_and_convert_sampling_grid_dims_in_pixels(params)
    params["device_name"] = \
        _check_and_convert_device_name(params)
    params["least_squares_alg_params"] = \
        _check_and_convert_least_squares_alg_params(params)

    for key in tuple(params.keys()):
        if key not in original_params:
            del params[key]
    
    return params



def _check_and_convert_ml_data_dict(params):
    params = params.copy()

    params["name_of_obj_alias_of_ml_data_dict"] = "ml_data_dict"
    params["ml_data_normalizer"] = _generate_default_ml_data_normalizer()
    params["target_numerical_data_container_cls"] = None
    params["target_device"] = None
    params["variable_axis_size_dict"] = None
    params["ml_data_shape_analyzer"] = _MLDataShapeAnalyzer()
    params["ml_data_type_validator"] = _MLDataTypeValidator()
    params["normalizable_elems_are_normalized"] = False
    params["ml_data_value_validator"] = _MLDataValueValidator()

    module_alias = emicroml.modelling._common
    func_alias = module_alias._check_and_convert_ml_data_dict
    ml_data_dict = func_alias(params)

    return ml_data_dict



def _ml_data_dict_to_distortion_models(ml_data_dict,
                                       sampling_grid_dims_in_pixels,
                                       device_name,
                                       least_squares_alg_params):
    try:
        distortion_centers = \
            ml_data_dict["distortion_centers"]
        ml_data_dict_keys_related_to_distortion_params = \
            _generate_keys_related_to_distortion_params()

        distortion_models = tuple()
        for cbed_pattern_idx, _ in enumerate(distortion_centers):
            cls_alias = distoptica.StandardCoordTransformParams
            kwargs = {"skip_validation_and_conversion": True}
            for key in ml_data_dict_keys_related_to_distortion_params:
                param_name = "center" if ("center" in key) else key[:-1]
                param_val = ml_data_dict[key][cbed_pattern_idx]
                param_val = (tuple(param_val.tolist())
                             if (("center" in key) or ("vector" in key))
                             else param_val.item())
                kwargs[param_name] = param_val
            standard_coord_transform_params = cls_alias(**kwargs)

            kwargs = {"standard_coord_transform_params": \
                      standard_coord_transform_params,
                      "sampling_grid_dims_in_pixels": \
                      sampling_grid_dims_in_pixels,
                      "device_name": \
                      device_name,
                      "least_squares_alg_params": \
                      least_squares_alg_params,
                      "skip_validation_and_conversion": \
                      True}
            func_alias = distoptica.generate_standard_distortion_model
            distortion_model = func_alias(**kwargs)
            distortion_models += (distortion_model,)
    except:
        cbed_pattern_images = ml_data_dict["cbed_pattern_images"]
        num_cbed_patterns = cbed_pattern_images.shape[0]
        distortion_models = (None,) * num_cbed_patterns

    return distortion_models



def _check_and_convert_ml_data_dict_to_signals_params(params):
    original_params = params
    params = params.copy()

    kwargs = {"obj": params["ml_data_dict"], "obj_name": "ml_data_dict"}
    ml_data_dict = czekitout.convert.to_dict(**kwargs)

    device_name = params["device_name"]

    params = {"cbed_pattern_images": \
              ml_data_dict.get("cbed_pattern_images", None),
              "name_of_obj_alias_of_cbed_pattern_images": \
              "ml_data_dict['cbed_pattern_images']",
              "target_device": \
              _get_device(device_name),
              **params}
    cbed_pattern_images = _check_and_convert_cbed_pattern_images(params)
    params["ml_data_dict"]["cbed_pattern_images"] = cbed_pattern_images

    params["data_chunk_dims_are_to_be_expanded_temporarily"] = \
        False
    params["expected_ml_data_dict_keys"] = \
        ("cbed_pattern_images",)
    params["ml_data_dict"] = \
        _check_and_convert_ml_data_dict(params)
    params["sampling_grid_dims_in_pixels"] = \
        _check_and_convert_sampling_grid_dims_in_pixels(params)
    params["device_name"] = \
        _check_and_convert_device_name(params)
    params["least_squares_alg_params"] = \
        _check_and_convert_least_squares_alg_params(params)

    for key in tuple(params.keys()):
        if key not in original_params:
            del params[key]
    
    return params



def _check_and_convert_cbed_pattern_images(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_cbed_pattern_images"
    func_alias = getattr(module_alias, func_name)
    cbed_pattern_images = func_alias(**kwargs)

    return cbed_pattern_images



def _min_max_normalize_image_stack(image_stack):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_min_max_normalize_image_stack"
    func_alias = getattr(module_alias, func_name)
    normalized_image_stack = func_alias(**kwargs)

    return normalized_image_stack



def _ml_data_dict_to_signals(ml_data_dict,
                             sampling_grid_dims_in_pixels,
                             device_name,
                             least_squares_alg_params):
    kwargs = \
        locals()
    distortion_models = \
        _ml_data_dict_to_distortion_models(**kwargs)

    kwargs = \
        {"ml_data_dict": ml_data_dict, "device_name": device_name}
    cbed_pattern_images = \
        _get_cbed_pattern_images_from_ml_data_dict(**kwargs)

    kwargs = \
        {"ml_data_dict": ml_data_dict}
    disk_overlap_maps = \
        _get_disk_overlap_maps_from_ml_data_dict(**kwargs)
    undistorted_disk_sets = \
        _generate_undistorted_disk_sets_from_ml_data_dict(**kwargs)
    mask_frames = \
        _calc_mask_frames_from_cbed_pattern_images(**kwargs)

    signals = tuple()
    global_symbol_table = globals()
    for cbed_pattern_idx, _ in enumerate(cbed_pattern_images):
        func_name = ("_construct_cbed_pattern_signal"
                     "_using_objs_extracted_from_ml_data_dict")
        func_alias = global_symbol_table[func_name]
        kwargs = {"undistorted_disk_set": \
                  undistorted_disk_sets[cbed_pattern_idx],
                  "cbed_pattern_image": \
                  cbed_pattern_images[cbed_pattern_idx],
                  "distortion_model": \
                  distortion_models[cbed_pattern_idx],
                  "mask_frame": \
                  mask_frames[cbed_pattern_idx],
                  "disk_overlap_map": \
                  disk_overlap_maps[cbed_pattern_idx]}
        cbed_pattern_signal = func_alias(**kwargs)
        signals += (cbed_pattern_signal,)

    return signals



def _get_cbed_pattern_images_from_ml_data_dict(ml_data_dict, device_name):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_get_cbed_pattern_images_from_ml_data_dict"
    func_alias = getattr(module_alias, func_name)
    cbed_pattern_images = func_alias(**kwargs)

    return cbed_pattern_images



def _get_disk_overlap_maps_from_ml_data_dict(ml_data_dict):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_get_disk_overlap_maps_from_ml_data_dict"
    func_alias = getattr(module_alias, func_name)
    disk_overlap_maps = func_alias(**kwargs)

    return disk_overlap_maps



def _generate_undistorted_disk_sets_from_ml_data_dict(ml_data_dict):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_generate_undistorted_disk_sets_from_ml_data_dict"
    func_alias = getattr(module_alias, func_name)
    undistorted_disk_sets = func_alias(**kwargs)

    return undistorted_disk_sets



def _calc_mask_frames_from_cbed_pattern_images(ml_data_dict):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_calc_mask_frames_from_cbed_pattern_images"
    func_alias = getattr(module_alias, func_name)
    mask_frames = func_alias(**kwargs)

    return mask_frames



def _construct_cbed_pattern_signal_using_objs_extracted_from_ml_data_dict(
        undistorted_disk_set,
        cbed_pattern_image,
        distortion_model,
        mask_frame,
        disk_overlap_map):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = ("_construct_cbed_pattern_signal_using_objs"
                 "_extracted_from_ml_data_dict")
    func_alias = getattr(module_alias, func_name)
    cbed_pattern_signal = func_alias(**kwargs)

    return cbed_pattern_signal



_module_alias = \
    emicroml.modelling.cbed._common
_default_entire_ml_dataset_is_to_be_cached = \
    _module_alias._default_entire_ml_dataset_is_to_be_cached
_default_ml_data_values_are_to_be_checked = \
    _module_alias._default_ml_data_values_are_to_be_checked
_default_max_num_ml_data_instances_per_chunk = \
    _module_alias._default_max_num_ml_data_instances_per_chunk
_default_single_dim_slice = \
    _module_alias._default_single_dim_slice



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataset
class _MLDataset(_cls_alias):
    def __init__(self,
                 path_to_ml_dataset,
                 entire_ml_dataset_is_to_be_cached,
                 ml_data_values_are_to_be_checked,
                 max_num_ml_data_instances_per_chunk,
                 skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLDataset
        cls_alias.__init__(self, ctor_params)
        
        return None



    def execute_post_core_attrs_update_actions(self):
        super().execute_post_core_attrs_update_actions()

        self_core_attrs = self.get_core_attrs(deep_copy=False)

        func_alias = _get_num_pixels_across_each_cbed_pattern
        kwargs = {"path_to_ml_dataset": \
                  self_core_attrs["path_to_ml_dataset"],
                  "ml_data_shape_analyzer": \
                  _MLDataShapeAnalyzer()}
        self._num_pixels_across_each_cbed_pattern = func_alias(**kwargs)

        func_alias = _get_max_num_disks_in_any_cbed_pattern
        self._max_num_disks_in_any_cbed_pattern = func_alias(**kwargs)

        return None



    def _generate_ml_data_normalization_weights_and_biases_loader(self):
        super()._generate_ml_data_normalization_weights_and_biases_loader()

        self_core_attrs = self.get_core_attrs(deep_copy=False)
        
        kwargs = \
            {"max_num_ml_data_instances_per_file_update": \
             self_core_attrs["max_num_ml_data_instances_per_chunk"]}
        ml_data_normalization_weights_and_biases_loader = \
            _MLDataNormalizationWeightsAndBiasesLoader(**kwargs)

        return ml_data_normalization_weights_and_biases_loader



    def _generate_torch_ml_dataset(self):
        super()._generate_torch_ml_dataset()

        self_core_attrs = \
            self.get_core_attrs(deep_copy=False)
        ml_data_normalization_weights_and_biases_loader = \
            self._generate_ml_data_normalization_weights_and_biases_loader()

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._TorchMLDataset
        kwargs = {"path_to_ml_dataset": \
                  self_core_attrs["path_to_ml_dataset"],
                  "ml_data_normalization_weights_and_biases_loader": \
                  ml_data_normalization_weights_and_biases_loader,
                  "ml_data_type_validator": \
                  _MLDataTypeValidator(),
                  "ml_data_shape_analyzer": \
                  _MLDataShapeAnalyzer(),
                  "entire_ml_dataset_is_to_be_cached": \
                  self_core_attrs["entire_ml_dataset_is_to_be_cached"],
                  "ml_data_values_are_to_be_checked": \
                  self_core_attrs["ml_data_values_are_to_be_checked"],
                  "ml_data_dict_elem_decoders": \
                  _generate_ml_data_dict_elem_decoders(),
                  "ml_data_dict_elem_decoding_order": \
                  _generate_ml_data_dict_elem_decoding_order()}
        torch_ml_dataset = cls_alias(**kwargs)

        return torch_ml_dataset



    def get_ml_data_instances_as_signals(
            self,
            single_dim_slice=\
            _default_single_dim_slice,
            device_name=\
            _default_device_name,
            sampling_grid_dims_in_pixels=\
            _default_sampling_grid_dims_in_pixels,
            least_squares_alg_params=\
            _default_least_squares_alg_params):
        r"""Return a subset of the machine learning data instances as a sequence
        of Hyperspy signals.

        See the documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern`,
        :class:`distoptica.DistortionModel`, and
        :class:`hyperspy._signals.signal2d.Signal2D` for discussions on "fake"
        CBED patterns, distortion models, and Hyperspy signals respectively.

        For each machine learning (ML) data instance in the subset, an instance
        ``distortion_model`` of the class :class:`distoptica.DistortionModel` is
        constructed according to the ML data instance's features. The object
        ``distortion_model`` is a distortion model that describes the distortion
        field of the imaged CBED pattern of the ML data instance. After
        constructing ``distortion_model``, an instance
        :class:`fakecbed.discretized.CBEDPattern` is constructed according to
        the ML data instance's features and
        ``distortion_model``. ``fake_cbed_pattern`` is a fake CBED pattern
        representation of the CBED pattern of the ML data instance. Next, a
        Hyperspy signal ``fake_cbed_pattern_signal`` is obtained from
        ``fake_cbed_pattern.signal``. The Hyperspy signal representation of the
        ML data instance is obtained by modifying in place
        ``fake_cbed_pattern_signal.data[1:3]`` according to the ML data
        instance's features. Note that the illumination support of the fake CBED
        pattern representation of the CBED pattern of the ML data instance is
        inferred from the features of the ML data instance, and is stored in
        ``fake_cbed_pattern_signal.data[1]``. Moreover, the illumination suport
        implied by the signal's metadata should be ignored.

        Parameters
        ----------
        single_dim_slice : `int` | `array_like` (`int`, ndim=1)  | `slice`, optional
            ``single_dim_slice`` specifies the subset of ML data instances to
            return as a dictionary. The ML data instances are indexed from ``0``
            to ``total_num_ml_data_instances-1``, where
            ``total_num_ml_data_instances`` is the total number of ML data
            instances in the ML dataset.
            ``tuple(range(total_num_ml_data_instances))[single_dim_slice]``
            yields the indices ``ml_data_instance_subset_indices`` of the ML 
            data instances to return.
        device_name : `str` | `None`, optional
            This parameter specifies the device to be used to perform
            computationally intensive calls to PyTorch functions and to store
            intermediate arrays of the type :class:`torch.Tensor`. If
            ``device_name`` is a string, then it is the name of the device to be
            used, e.g. ``”cuda”`` or ``”cpu”``. If ``device_name`` is set to
            ``None`` and a GPU device is available, then a GPU device is to be
            used. Otherwise, the CPU is used.
        sampling_grid_dims_in_pixels : `array_like` (`int`, shape=(2,)), optional
            The dimensions of the sampling grid, in units of pixels, used for
            all distortion models.
        least_squares_alg_params : :class:`distoptica.LeastSquaresAlgParams` | `None`, optional
            ``least_squares_alg_params`` specifies the parameters of the
            least-squares algorithm to be used to calculate the mappings of
            fractional Cartesian coordinates of distorted images to those of the
            corresponding undistorted images. ``least_squares_alg_params`` is
            used to calculate the interim distortion models mentioned above in
            the summary documentation. If ``least_squares_alg_params`` is set to
            ``None``, then the parameter will be reassigned to the value
            ``distoptica.LeastSquaresAlgParams()``. See the documentation for
            the class :class:`distoptica.LeastSquaresAlgParams` for details on
            the parameters of the least-squares algorithm.

        Returns
        -------
        ml_data_instances_as_signals : `array_like` (:class:`hyperspy._signals.signal2d.Signal2D`, ndim=1)
            The subset of ML data instances, represented as a sequence of
            Hyperspy signals. Let ``num_ml_data_instances_in_subset`` be
            ``len(ml_data_instances_as_signals)``. For every nonnegative integer
            ``n`` less than ``num_ml_data_instances_in_subset``, then
            ``ml_data_instances_as_signals[n]`` yields the Hyperspy signal 
            representation of the ML data instance with the index
            ``ml_data_instance_subset_indices[n]``.

        """
        params = \
            {key: val
             for key, val in locals().items()
             if (key not in ("self", "__class__"))}
        sampling_grid_dims_in_pixels = \
            _check_and_convert_sampling_grid_dims_in_pixels(params)
        least_squares_alg_params = \
            _check_and_convert_least_squares_alg_params(params)

        kwargs = {"single_dim_slice": single_dim_slice,
                  "device_name": device_name,
                  "decode": True,
                  "unnormalize_normalizable_elems": True}
        ml_data_instances = self.get_ml_data_instances(**kwargs)

        kwargs = \
            {"ml_data_dict": ml_data_instances,
             "sampling_grid_dims_in_pixels": sampling_grid_dims_in_pixels,
             "device_name": device_name,
             "least_squares_alg_params": least_squares_alg_params}
        ml_data_instances_as_signals = \
            _ml_data_dict_to_signals(**kwargs)

        return ml_data_instances_as_signals



    @property
    def num_pixels_across_each_cbed_pattern(self):
        r"""`int`: The number of pixels across each imaged CBED pattern stored 
        in the machine learning dataset.

        Note that ``num_pixels_across_each_cbed_pattern`` should be considered
        **read-only**.

        """
        result = self._num_pixels_across_each_cbed_pattern
        
        return result



    @property
    def max_num_disks_in_any_cbed_pattern(self):
        r"""`int`: The maximum possible number of CBED disks in any imaged CBED 
        pattern stored in the machine learning dataset.

        Note that ``max_num_disks_in_any_cbed_pattern`` should be considered
        **read-only**.

        """
        result = self._max_num_disks_in_any_cbed_pattern

        return result



def _check_and_convert_ml_training_dataset(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_ml_training_dataset
    ml_training_dataset = func_alias(params)

    return ml_training_dataset



def _pre_serialize_ml_training_dataset(ml_training_dataset):
    obj_to_pre_serialize = ml_training_dataset
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._pre_serialize_ml_training_dataset
    serializable_rep = func_alias(obj_to_pre_serialize)
    
    return serializable_rep



_module_alias = emicroml.modelling.cbed._common
_default_ml_training_dataset = _module_alias._default_ml_training_dataset



def _check_and_convert_ml_validation_dataset(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_ml_validation_dataset
    ml_validation_dataset = func_alias(params)

    return ml_validation_dataset



def _pre_serialize_ml_validation_dataset(ml_validation_dataset):
    obj_to_pre_serialize = ml_validation_dataset
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._pre_serialize_ml_validation_dataset
    serializable_rep = func_alias(obj_to_pre_serialize)
    
    return serializable_rep



_module_alias = emicroml.modelling.cbed._common
_default_ml_validation_dataset = _module_alias._default_ml_validation_dataset



def _check_and_convert_ml_testing_dataset(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_ml_testing_dataset
    ml_testing_dataset = func_alias(params)

    return ml_testing_dataset



def _pre_serialize_ml_testing_dataset(ml_testing_dataset):
    obj_to_pre_serialize = ml_testing_dataset
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._pre_serialize_ml_testing_dataset
    serializable_rep = func_alias(obj_to_pre_serialize)
    
    return serializable_rep



_module_alias = \
    emicroml.modelling.cbed._common
_default_ml_testing_dataset = \
    _module_alias._default_ml_testing_dataset
_default_mini_batch_size = \
    _module_alias._default_mini_batch_size
_default_num_data_loader_workers = \
    _module_alias._default_num_data_loader_workers



_module_alias = emicroml.modelling.cbed._common
_cls_alias = _module_alias._MLDatasetManager
class _MLDatasetManager(_cls_alias):
    def __init__(self,
                 ml_training_dataset,
                 ml_validation_dataset,
                 ml_testing_dataset,
                 mini_batch_size,
                 rng_seed,
                 num_data_loader_workers,
                 skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling.cbed._common
        cls_alias = module_alias._MLDatasetManager
        cls_alias.__init__(self, ctor_params)

        return None



def _check_and_convert_ml_dataset_manager(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_ml_dataset_manager
    ml_dataset_manager = func_alias(params)

    return ml_dataset_manager



def _pre_serialize_ml_dataset_manager(ml_dataset_manager):
    obj_to_pre_serialize = ml_dataset_manager
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._pre_serialize_ml_dataset_manager
    serializable_rep = func_alias(obj_to_pre_serialize)
    
    return serializable_rep



class _DistopticaNet(torch.nn.Module):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 mini_batch_norm_eps):
        super().__init__()

        self._num_pixels_across_each_cbed_pattern = \
            num_pixels_across_each_cbed_pattern
        self._mini_batch_norm_eps = \
            mini_batch_norm_eps
        
        self._distoptica_net = self._generate_distoptica_net()

        return None



    def _generate_distoptica_net(self):
        num_filters_in_first_conv_layer = \
            64
        building_block_counts_in_stages = \
            _building_block_counts_in_stages_of_distoptica_net
        num_downsamplings = \
            len(building_block_counts_in_stages)
        num_nodes_in_second_last_layer = \
            (num_filters_in_first_conv_layer * (2**num_downsamplings))

        module_alias = emicroml.modelling._common
        kwargs = {"num_input_channels": \
                  1,
                  "num_filters_in_first_conv_layer": \
                  num_filters_in_first_conv_layer,
                  "kernel_size_of_first_conv_layer": \
                  7,
                  "max_kernel_size_of_resnet_building_blocks": \
                  3,
                  "building_block_counts_in_stages": \
                  building_block_counts_in_stages,
                  "height_of_input_tensor_in_pixels": \
                  self._num_pixels_across_each_cbed_pattern,
                  "width_of_input_tensor_in_pixels": \
                  self._num_pixels_across_each_cbed_pattern,
                  "num_nodes_in_second_last_layer": \
                  num_nodes_in_second_last_layer,
                  "num_nodes_in_last_layer": \
                  8,
                  "mini_batch_norm_eps": \
                  self._mini_batch_norm_eps}
        distoptica_net = module_alias._DistopticaNet(**kwargs)

        return distoptica_net



    def forward(self, ml_inputs):
        enhanced_cbed_pattern_images = \
            self._get_and_enhance_cbed_pattern_images(ml_inputs)

        intermediate_tensor = enhanced_cbed_pattern_images
        intermediate_tensor, _ = self._distoptica_net(intermediate_tensor)

        ml_predictions = dict()
        keys = ("quadratic_radial_distortion_amplitudes",
                "spiral_distortion_amplitudes",
                "elliptical_distortion_vectors",
                "parabolic_distortion_vectors",
                "distortion_centers")

        stop = 0

        for key_idx, key in enumerate(keys):
            start = stop
            stop = start + 1 + ("amplitudes" not in key)

            multi_dim_slice = ((slice(None), start)
                                if ("amplitudes" in key)
                                else (slice(None), slice(start, stop)))
            
            output_tensor = intermediate_tensor[multi_dim_slice]
            ml_predictions[key] = output_tensor

        return ml_predictions



    def _get_and_enhance_cbed_pattern_images(self, ml_inputs):
        kwargs = {"image_stack": ml_inputs["cbed_pattern_images"]}
        enhanced_cbed_pattern_images = _min_max_normalize_image_stack(**kwargs)

        gamma = 0.3

        enhanced_cbed_pattern_images = \
            torch.unsqueeze(enhanced_cbed_pattern_images, dim=1)
        enhanced_cbed_pattern_images = \
            torch.pow(enhanced_cbed_pattern_images, gamma)
        enhanced_cbed_pattern_images = \
            kornia.enhance.equalize(enhanced_cbed_pattern_images)

        kwargs = {"input": enhanced_cbed_pattern_images,
                  "min": 0,
                  "max": 1}
        enhanced_cbed_pattern_images = torch.clip(**kwargs)

        return enhanced_cbed_pattern_images



def _check_and_convert_architecture(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_architecture
    architecture = func_alias(params)

    return architecture



def _check_and_convert_mini_batch_norm_eps(params):
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._check_and_convert_mini_batch_norm_eps
    mini_batch_norm_eps = func_alias(params)

    return mini_batch_norm_eps



def _check_and_convert_normalization_weights(params):
    obj_name = "normalization_weights"

    global_symbol_table = globals()

    params = params.copy()

    func_name = ("_generate_default"
                 "_ml_data_normalization_weights_and_biases_loader")
    func_alias = global_symbol_table[func_name]
    params["ml_data_normalization_weights_and_biases_loader"] = func_alias()

    func_name = "_generate_default_normalization_weights"
    func_alias = global_symbol_table[func_name]
    params["default_normalization_weights"] = func_alias()

    module_alias = emicroml.modelling._common
    func_alias = module_alias._check_and_convert_normalization_weights
    normalization_weights = func_alias(params)

    return normalization_weights



def _generate_default_normalization_weights():
    obj_name = "normalization_weights"

    ml_data_normalizer = _generate_default_ml_data_normalizer()
    extrema_cache = ml_data_normalizer._extrema_cache

    kwargs = {"extrema_cache": extrema_cache}
    _update_extrema_cache_for_default_normalization_weights_and_biases(**kwargs)

    ml_data_normalizer._update_normalization_weights_and_biases()

    attr_name = "_" + obj_name
    normalization_weights = getattr(ml_data_normalizer, attr_name)

    return normalization_weights



def _update_extrema_cache_for_default_normalization_weights_and_biases(
        extrema_cache):
    kwargs = {"reference_pt": \
              _default_reference_pt,
              "rng_seed": \
              _default_rng_seed,
              "sampling_grid_dims_in_pixels": \
              _default_sampling_grid_dims_in_pixels,
              "least_squares_alg_params": \
              _default_least_squares_alg_params,
              "device_name": \
              _default_device_name,
              "skip_validation_and_conversion": \
              True}
    distortion_model_generator = _DefaultDistortionModelGenerator(**kwargs)

    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key_1 in all_valid_ml_data_dict_keys:
        if key_1 not in extrema_cache:
            continue
        for key_2 in ("min", "max"):
                key_3 = "_" + key_1[:-1] + "_" + key_2
                
                obj_from_which_to_get_attr = distortion_model_generator
                attr_name = key_3
                default_value_if_attr_does_not_exist = int(key_2 == "max")
                
                args = (obj_from_which_to_get_attr,
                        attr_name,
                        default_value_if_attr_does_not_exist)
                extrema_cache[key_1][key_2] = getattr(*args)

    return None



def _check_and_convert_normalization_biases(params):
    obj_name = "normalization_biases"

    global_symbol_table = globals()

    params = params.copy()

    func_name = ("_generate_default"
                 "_ml_data_normalization_weights_and_biases_loader")
    func_alias = global_symbol_table[func_name]
    params["ml_data_normalization_weights_and_biases_loader"] = func_alias()

    func_name = "_generate_default_normalization_biases"
    func_alias = global_symbol_table[func_name]
    params["default_normalization_biases"] = func_alias()

    module_alias = emicroml.modelling._common
    func_alias = module_alias._check_and_convert_normalization_biases
    normalization_biases = func_alias(params)

    return normalization_biases



def _generate_default_normalization_biases():
    obj_name = "normalization_biases"

    ml_data_normalizer = _generate_default_ml_data_normalizer()
    extrema_cache = ml_data_normalizer._extrema_cache

    kwargs = {"extrema_cache": extrema_cache}
    _update_extrema_cache_for_default_normalization_weights_and_biases(**kwargs)

    ml_data_normalizer._update_normalization_weights_and_biases()

    attr_name = "_" + obj_name
    normalization_biases = getattr(ml_data_normalizer, attr_name)

    return normalization_biases



def _get_device_name(device):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._get_device_name
    device_name = func_alias(**kwargs)

    return device_name



_module_alias = \
    emicroml.modelling.cbed._common
_default_architecture = \
    "distoptica_net"
_default_mini_batch_norm_eps = \
    _module_alias._default_mini_batch_norm_eps
_default_normalization_weights = \
    _module_alias._default_normalization_weights
_default_normalization_biases = \
    _module_alias._default_normalization_biases
_default_normalizable_elems_of_ml_inputs_are_normalized = \
    _module_alias._default_normalizable_elems_of_ml_inputs_are_normalized
_default_unnormalize_normalizable_elems_of_ml_predictions = \
    _module_alias._default_unnormalize_normalizable_elems_of_ml_predictions



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLModel
class _MLModel(_cls_alias):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 max_num_disks_in_any_cbed_pattern,
                 architecture,
                 mini_batch_norm_eps,
                 normalization_weights,
                 normalization_biases):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}

        kwargs = \
            {"num_pixels_across_each_cbed_pattern": \
             num_pixels_across_each_cbed_pattern,
             "max_num_disks_in_any_cbed_pattern": \
             max_num_disks_in_any_cbed_pattern}
        variable_axis_size_dict = \
            self._generate_variable_axis_size_dict(**kwargs)
        
        expected_keys_of_ml_inputs = \
            self._generate_expected_keys_of_ml_inputs()

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLModel
        kwargs = {"ml_data_normalizer": _generate_default_ml_data_normalizer(),
                  "ml_data_type_validator": _MLDataTypeValidator(),
                  "ml_data_value_validator": _MLDataValueValidator(),
                  "ml_data_shape_analyzer": _MLDataShapeAnalyzer(),
                  "variable_axis_size_dict": variable_axis_size_dict,
                  "expected_keys_of_ml_inputs": expected_keys_of_ml_inputs,
                  "subcls_ctor_params": ctor_params}
        cls_alias.__init__(self, **kwargs)

        self._initialize_ml_model_cmpnts(architecture,
                                         num_pixels_across_each_cbed_pattern,
                                         mini_batch_norm_eps)

        return None



    def _generate_variable_axis_size_dict(self,
                                          num_pixels_across_each_cbed_pattern,
                                          max_num_disks_in_any_cbed_pattern):
        variable_axis_size_dict_keys = _generate_variable_axis_size_dict_keys()
        num_keys = len(variable_axis_size_dict_keys)

        variable_axis_size_dict = dict()
        for key_idx, key in enumerate(variable_axis_size_dict_keys):
            if key_idx == 0:
                variable_size_of_axis = None
            elif key_idx == num_keys-1:
                variable_size_of_axis = max_num_disks_in_any_cbed_pattern
            else:
                variable_size_of_axis = num_pixels_across_each_cbed_pattern
                
            variable_axis_size_dict[key] = variable_size_of_axis

        return variable_axis_size_dict



    def _generate_expected_keys_of_ml_inputs(self):
        expected_keys_of_ml_inputs = ("cbed_pattern_images",)

        return expected_keys_of_ml_inputs



    def _check_and_convert_ctor_params(self, ctor_params):
        ctor_params = ctor_params.copy()

        global_symbol_table = globals()
        for ctor_param_name in ctor_params.keys():
            func_name = "_check_and_convert_" + ctor_param_name
            func_alias = global_symbol_table[func_name]
            ctor_params[ctor_param_name] = func_alias(params=ctor_params)

        return ctor_params



    def _initialize_ml_model_cmpnts(self,
                                    architecture,
                                    num_pixels_across_each_cbed_pattern,
                                    mini_batch_norm_eps):
        base_model_cls = _DistopticaNet

        self._base_model = base_model_cls(num_pixels_across_each_cbed_pattern,
                                          mini_batch_norm_eps)

        return None



    def forward(self, ml_inputs):
        ml_predictions = self._base_model(ml_inputs)

        return ml_predictions



    def make_predictions(
            self,
            ml_inputs,
            unnormalize_normalizable_elems_of_ml_predictions=\
            _default_unnormalize_normalizable_elems_of_ml_predictions):
        kwargs = {"obj": ml_inputs, "obj_name": "ml_inputs"}
        ml_inputs = czekitout.convert.to_dict(**kwargs)

        params = {"cbed_pattern_images": \
                  ml_inputs.get("cbed_pattern_images", None),
                  "name_of_obj_alias_of_cbed_pattern_images": \
                  "ml_inputs['cbed_pattern_images']",
                  "target_device": \
                  next(self.parameters()).device}
        cbed_pattern_images = _check_and_convert_cbed_pattern_images(params)
        ml_inputs["cbed_pattern_images"] = cbed_pattern_images

        kwargs = {"ml_inputs": \
                  ml_inputs,
                  "unnormalize_normalizable_elems_of_ml_predictions": \
                  unnormalize_normalizable_elems_of_ml_predictions,
                  "normalizable_elems_of_ml_inputs_are_normalized": \
                  True}
        ml_predictions = super().make_predictions(**kwargs)

        return ml_predictions



    def predict_distortion_models(self,
                                  cbed_pattern_images,
                                  sampling_grid_dims_in_pixels=\
                                  _default_sampling_grid_dims_in_pixels,
                                  least_squares_alg_params=\
                                  _default_least_squares_alg_params):
        params = \
            {key: val
             for key, val in locals().items()
             if (key not in ("self", "__class__"))}
        sampling_grid_dims_in_pixels = \
            _check_and_convert_sampling_grid_dims_in_pixels(params)
        least_squares_alg_params = \
            _check_and_convert_least_squares_alg_params(params)

        kwargs ={"ml_inputs": {"cbed_pattern_images": cbed_pattern_images},
                 "unnormalize_normalizable_elems_of_ml_predictions": True}
        ml_predictions = self.make_predictions(**kwargs)

        device = next(self.parameters()).device

        kwargs = {"ml_data_dict": ml_predictions,
                  "sampling_grid_dims_in_pixels": sampling_grid_dims_in_pixels,
                  "least_squares_alg_params": least_squares_alg_params,
                  "device_name": _get_device_name(device)}
        distortion_models = _ml_data_dict_to_distortion_models(**kwargs)

        return distortion_models



def _calc_shifted_q(ml_data_dict, ml_model):
    kwargs = \
        locals()
    distortion_model_set_params = \
        _generate_distortion_model_set_params(**kwargs)

    distortion_centers = distortion_model_set_params["distortion_centers"]

    kwargs = \
        {"ml_model": ml_model, "distortion_centers": distortion_centers}
    cached_objs_of_coord_transform_set = \
        _calc_cached_objs_of_coord_transform_set(**kwargs)

    kwargs = {"cached_objs_of_coord_transform_set": \
              cached_objs_of_coord_transform_set,
              "distortion_model_set_params": \
              distortion_model_set_params}
    q = _calc_q(**kwargs)

    shifted_q = q
    shifted_q[:, :] -= q.mean(dim=(2, 3))[:, :, None, None]

    return shifted_q



def _generate_distortion_model_set_params(ml_data_dict, ml_model):
    key_subset = _generate_keys_related_to_distortion_params()

    distortion_model_set_params = {key: 1.0*ml_data_dict[key]
                                   for key
                                   in key_subset}

    kwargs = {"ml_data_dict": distortion_model_set_params,
              "normalization_weights": ml_model._normalization_weights,
              "normalization_biases": ml_model._normalization_biases}
    _unnormalize_normalizable_elems_in_ml_data_dict(**kwargs)

    return distortion_model_set_params



def _calc_cached_objs_of_coord_transform_set(ml_model, distortion_centers):
    kwargs = locals()
    u_x, u_y = _calc_u_x_and_u_y(**kwargs)

    x_c_D = distortion_centers[:, 0]
    y_c_D = distortion_centers[:, 1]

    u_r_cos_of_u_theta = u_x[:, :, :] - x_c_D[:, None, None]
    u_r_sin_of_u_theta = u_y[:, :, :] - y_c_D[:, None, None]
    
    u_r_sq = (u_r_cos_of_u_theta*u_r_cos_of_u_theta
              + u_r_sin_of_u_theta*u_r_sin_of_u_theta)

    u_r_sq_cos_of_2_u_theta = (u_r_cos_of_u_theta*u_r_cos_of_u_theta
                               - u_r_sin_of_u_theta*u_r_sin_of_u_theta)
    u_r_sq_sin_of_2_u_theta = 2*u_r_cos_of_u_theta*u_r_sin_of_u_theta

    cached_objs_of_coord_transform_set = {"u_x": \
                                          u_x,
                                          "u_y": \
                                          u_y,
                                          "u_r_cos_of_u_theta": \
                                          u_r_cos_of_u_theta,
                                          "u_r_sin_of_u_theta": \
                                          u_r_sin_of_u_theta,
                                          "u_r_sq": \
                                          u_r_sq,
                                          "u_r_sq_cos_of_2_u_theta": \
                                          u_r_sq_cos_of_2_u_theta,
                                          "u_r_sq_sin_of_2_u_theta": \
                                          u_r_sq_sin_of_2_u_theta}

    return cached_objs_of_coord_transform_set



def _calc_u_x_and_u_y(ml_model, distortion_centers):
    sampling_grid_dims_in_pixels = \
        2*(ml_model._base_model._num_pixels_across_each_cbed_pattern,)
    device = \
        distortion_centers.device
    mini_batch_size = \
        distortion_centers.shape[0]

    j_range = torch.arange(sampling_grid_dims_in_pixels[0], device=device)
    i_range = torch.arange(sampling_grid_dims_in_pixels[1], device=device)
        
    pair_of_1d_coord_arrays = ((j_range + 0.5) / j_range.numel(),
                               1 - (i_range + 0.5) / i_range.numel())
    sampling_grid = torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")
        
    u_x_shape = (mini_batch_size,) + sampling_grid[0].shape
    u_x = torch.zeros(u_x_shape,
                      dtype=distortion_centers.dtype,
                      device=distortion_centers.device)
    
    u_y = torch.zeros_like(u_x)

    for ml_data_instance_idx in range(mini_batch_size):
        u_x[ml_data_instance_idx] = sampling_grid[0]
        u_y[ml_data_instance_idx] = sampling_grid[1]

    return u_x, u_y



def _calc_q(cached_objs_of_coord_transform_set, distortion_model_set_params):
    u_x = cached_objs_of_coord_transform_set["u_x"]
    u_y = cached_objs_of_coord_transform_set["u_y"]

    q_shape = (u_x.shape[0], 2) + u_x.shape[1:]
    q = torch.zeros(q_shape, dtype=u_x.dtype, device=u_x.device)

    q[:, 0] = u_x
    q[:, 1] = u_y

    kwargs = {"cached_objs_of_coord_transform_set": \
              cached_objs_of_coord_transform_set,
              "distortion_model_set_params": \
              distortion_model_set_params,
              "q": \
              q}
    _add_quadratic_radial_and_spiral_distortion_fields_to_q(**kwargs)
    _add_parabolic_distortion_field_to_q(**kwargs)
    _add_elliptical_distortion_field_to_q(**kwargs)
    
    return q



def _add_quadratic_radial_and_spiral_distortion_fields_to_q(
        cached_objs_of_coord_transform_set, distortion_model_set_params, q):
    u_r_sq = \
        cached_objs_of_coord_transform_set["u_r_sq"]
    u_r_cos_of_u_theta = \
        cached_objs_of_coord_transform_set["u_r_cos_of_u_theta"]
    u_r_sin_of_u_theta = \
        cached_objs_of_coord_transform_set["u_r_sin_of_u_theta"]

    A_r_0_2 = \
        distortion_model_set_params["quadratic_radial_distortion_amplitudes"]
    A_t_0_2 = \
        distortion_model_set_params["spiral_distortion_amplitudes"]

    q[:, 0] += u_r_sq * (u_r_cos_of_u_theta*A_r_0_2[:, None, None]
                         - u_r_sin_of_u_theta*A_t_0_2[:, None, None])
    q[:, 1] += u_r_sq * (u_r_sin_of_u_theta*A_r_0_2[:, None, None]
                         + u_r_cos_of_u_theta*A_t_0_2[:, None, None])

    return None



def _add_parabolic_distortion_field_to_q(cached_objs_of_coord_transform_set,
                                         distortion_model_set_params,
                                         q):
    u_r_sq = \
        cached_objs_of_coord_transform_set["u_r_sq"]
    u_r_sq_cos_of_2_u_theta = \
        cached_objs_of_coord_transform_set["u_r_sq_cos_of_2_u_theta"]
    u_r_sq_sin_of_2_u_theta = \
        cached_objs_of_coord_transform_set["u_r_sq_sin_of_2_u_theta"]

    A_r_1_1 = \
        distortion_model_set_params["parabolic_distortion_vectors"][:, 0]
    B_r_0_1 = \
        distortion_model_set_params["parabolic_distortion_vectors"][:, 1]

    q[:, 0] += ((2.0*u_r_sq + u_r_sq_cos_of_2_u_theta)*A_r_1_1[:, None, None]
                + u_r_sq_sin_of_2_u_theta*B_r_0_1[:, None, None]) / 3.0
    q[:, 1] += ((2.0*u_r_sq - u_r_sq_cos_of_2_u_theta)*B_r_0_1[:, None, None]
                + u_r_sq_sin_of_2_u_theta*A_r_1_1[:, None, None]) / 3.0

    return None



def _add_elliptical_distortion_field_to_q(cached_objs_of_coord_transform_set,
                                          distortion_model_set_params,
                                          q):
    u_r_cos_of_u_theta = \
        cached_objs_of_coord_transform_set["u_r_cos_of_u_theta"]
    u_r_sin_of_u_theta = \
        cached_objs_of_coord_transform_set["u_r_sin_of_u_theta"]

    A_r_2_0 = \
        distortion_model_set_params["elliptical_distortion_vectors"][:, 0]
    B_r_1_0 = \
        distortion_model_set_params["elliptical_distortion_vectors"][:, 1]

    q[:, 0] += (u_r_cos_of_u_theta*A_r_2_0[:, None, None]
                + u_r_sin_of_u_theta*B_r_1_0[:, None, None])
    q[:, 1] += (-u_r_sin_of_u_theta*A_r_2_0[:, None, None]
                + u_r_cos_of_u_theta*B_r_1_0[:, None, None])

    return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLMetricCalculator
class _MLMetricCalculator(_cls_alias):
    def __init__(self):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLMetricCalculator
        kwargs = dict()
        cls_alias.__init__(self, **kwargs)

        return None



    def _calc_metrics_of_current_mini_batch(
            self,
            ml_inputs,
            ml_predictions,
            ml_targets,
            ml_model,
            ml_dataset_manager,
            mini_batch_indices_for_entire_training_session):
        kwargs = \
            {key: val
             for key, val in locals().items()
             if (key not in ("self", "__class__"))}
        metrics_of_current_mini_batch = \
            super()._calc_metrics_of_current_mini_batch(**kwargs)

        kwargs = {"ml_data_dict": ml_targets, "ml_model": ml_model}
        target_shifted_q = _calc_shifted_q(**kwargs)

        kwargs = {"ml_data_dict": ml_predictions, "ml_model": ml_model}
        predicted_shifted_q = _calc_shifted_q(**kwargs)

        method_alias = self._calc_epes_of_adjusted_distortion_fields
        kwargs = {"target_shifted_q": target_shifted_q,
                  "predicted_shifted_q": predicted_shifted_q}
        epes_of_adjusted_distortion_fields = method_alias(**kwargs)

        metrics_of_current_mini_batch = {"epes_of_adjusted_distortion_fields": \
                                         epes_of_adjusted_distortion_fields}

        return metrics_of_current_mini_batch



    def _calc_epes_of_adjusted_distortion_fields(self,
                                                 target_shifted_q,
                                                 predicted_shifted_q):
        calc_euclidean_distances = torch.linalg.vector_norm
        kwargs = {"x": target_shifted_q-predicted_shifted_q, "dim": 1}
        euclidean_distances = calc_euclidean_distances(**kwargs)

        epes = euclidean_distances.mean(dim=(1, 2))
        epes_of_adjusted_distortion_fields = epes

        return epes_of_adjusted_distortion_fields



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLLossCalculator
class _MLLossCalculator(_cls_alias):
    def __init__(self):
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLLossCalculator
        kwargs = dict()
        cls_alias.__init__(self, **kwargs)

        return None



    def _calc_losses_of_current_mini_batch(
            self,
            ml_inputs,
            ml_predictions,
            ml_targets,
            ml_model,
            ml_dataset_manager,
            phase,
            ml_metric_manager,
            mini_batch_indices_for_entire_training_session):
        kwargs = \
            {key: val
             for key, val in locals().items()
             if (key not in ("self", "__class__"))}
        losses_of_current_mini_batch = \
            super()._calc_losses_of_current_mini_batch(**kwargs)

        metrics_of_current_mini_batch = \
            ml_metric_manager._metrics_of_current_mini_batch

        losses_of_current_mini_batch = {"total": 0.0}

        key_set_1 = ("epes_of_adjusted_distortion_fields",)

        for key_1 in key_set_1:
            key_2 = \
                "total"
            losses_of_current_mini_batch[key_1] = \
                metrics_of_current_mini_batch[key_1].mean()
            losses_of_current_mini_batch[key_2] += \
                losses_of_current_mini_batch[key_1]

        return losses_of_current_mini_batch



_module_alias = \
    emicroml.modelling.cbed._common
_default_checkpoints = \
    _module_alias._default_checkpoints
_default_lr_scheduler_manager = \
    _module_alias._default_lr_scheduler_manager
_default_output_dirname = \
    _module_alias._default_output_dirname
_default_misc_model_training_metadata = \
    _module_alias._default_misc_model_training_metadata



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLModelTrainer
class _MLModelTrainer(_cls_alias):
    def __init__(self,
                 ml_dataset_manager,
                 device_name,
                 checkpoints,
                 lr_scheduler_manager,
                 output_dirname,
                 misc_model_training_metadata,
                 skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLModelTrainer
        cls_alias.__init__(self, ctor_params)

        return None


    
    def train_ml_model(self, ml_model, ml_model_param_groups):
        self._ml_metric_calculator = _MLMetricCalculator()
        self._ml_loss_calculator = _MLLossCalculator()

        kwargs = {"ml_model": ml_model,
                  "ml_model_param_groups": ml_model_param_groups}
        super().train_ml_model(**kwargs)

        return None



_module_alias = \
    emicroml.modelling.cbed._common
_default_misc_model_testing_metadata = \
    _module_alias._default_misc_model_testing_metadata



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLModelTester
class _MLModelTester(_cls_alias):
    def __init__(self,
                 ml_dataset_manager,
                 device_name,
                 output_dirname,
                 misc_model_testing_metadata,
                 skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLModelTester
        cls_alias.__init__(self, ctor_params)

        return None


    
    def test_ml_model(self, ml_model):
        self._ml_metric_calculator = _MLMetricCalculator()

        kwargs = {"ml_model": ml_model}
        super().test_ml_model(**kwargs)

        return None



###########################
## Define error messages ##
###########################

_check_and_convert_num_pixels_across_each_cbed_pattern_err_msg_1 = \
    ("The object ``num_pixels_across_each_cbed_pattern`` must be positive "
     "integer that is divisible {}.")

_unnormalized_ml_data_instance_generator_err_msg_1 = \
    ("The object ``cbed_pattern_generator`` must generate CBED patterns of "
     "consistent dimensions.")

_custom_value_checker_for_cbed_pattern_images_err_msg_1 = \
    ("The HDF5 dataset at the HDF5 path ``'{}'`` of the HDF5 file at the file "
     "path ``'{}'`` must contain only images that are normalized such that the "
     "minimum and maximum pixel values are equal to zero and unity "
     "respectively for each image.")
_custom_value_checker_for_cbed_pattern_images_err_msg_2 = \
    ("The object ``{}['{}']`` must contain only images that are normalized "
     "such that the minimum and maximum pixel values are equal to zero and "
     "unity respectively for each image.")
