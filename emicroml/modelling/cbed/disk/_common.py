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
:mod:`emicroml.modelling.cbed.disk`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For generating the alphabet.
import string



# For validating and converting objects. Also for getting fully qualified names
# of instances of classes.
import czekitout.check
import czekitout.convert
import czekitout.name

# For general array handling.
import numpy as np

# For defining classes that support enforced validation and updatability.
import fancytypes

# For generating fake CBED patterns.
import fakecbed

# For building neural network models.
import torch

# For generating distortion models.
import distoptica

# For closing HDF5 files.
import h5py
import h5pywrappers

# For creating hyperspy signals, axes, and markers.
import hyperspy.api as hs
import hyperspy.signals
import hyperspy.axes

# For image processing tools that can be integrated into deep learning models.
import kornia

# For performing multi-resolution analysis.
import pywt
import pytorch_wavelets



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



def _check_and_convert_sampling_grid_dims_in_pixels(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_sampling_grid_dims_in_pixels"
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



def _check_and_convert_device_name(params):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_device_name"
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



def _check_and_convert_num_pixels_across_each_cbed_pattern(params):
    params["divisor"] = _generate_divisor_1()
    
    kwargs = {"params": params}
    module_alias = emicroml.modelling.cbed._common
    func_name = "_check_and_convert_num_pixels_across_each_cbed_pattern"
    func_alias = getattr(module_alias, func_name)
    num_pixels_across_each_cbed_pattern = func_alias(**kwargs)

    del params["divisor"]

    return num_pixels_across_each_cbed_pattern



def _generate_divisor_1():
    divisor_1 = 1

    return divisor_1



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



def _check_and_convert_num_pixels_across_each_expected_cropping_window(params):
    obj_name = "num_pixels_across_each_expected_cropping_window"

    func_alias = czekitout.convert.to_positive_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    num_pixels_across_each_expected_cropping_window = func_alias(**kwargs)

    divisor = _generate_divisor_2()

    current_func_name = ("_check_and_convert"
                         "_num_pixels_across_each_expected_cropping_window")

    if num_pixels_across_each_expected_cropping_window % divisor != 0:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(divisor)
        raise ValueError(err_msg)

    return num_pixels_across_each_expected_cropping_window



def _generate_divisor_2():
    divisor_2 = 2

    return divisor_2



def _pre_serialize_num_pixels_across_each_expected_cropping_window(
        num_pixels_across_each_expected_cropping_window):
    obj_to_pre_serialize = num_pixels_across_each_expected_cropping_window
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_num_pixels_across_each_expected_cropping_window(
        serializable_rep):
    num_pixels_across_each_expected_cropping_window = serializable_rep

    return num_pixels_across_each_expected_cropping_window



_module_alias = \
    emicroml.modelling.cbed._common
_default_num_pixels_across_each_cbed_pattern = \
    _module_alias._default_num_pixels_across_each_cbed_pattern
_default_max_num_disks_in_any_cbed_pattern = \
    20
_default_num_pixels_across_each_expected_cropping_window = \
    _default_num_pixels_across_each_cbed_pattern//4



_module_alias = emicroml.modelling.cbed._common
_cls_alias = _module_alias._DefaultCBEDPatternGenerator
class _DefaultCBEDPatternGenerator(_cls_alias):
    _validation_and_conversion_funcs_ = \
        {**_cls_alias._validation_and_conversion_funcs_,
         "num_pixels_across_each_cbed_pattern": \
         _check_and_convert_num_pixels_across_each_cbed_pattern,
         "num_pixels_across_each_expected_cropping_window": \
         _check_and_convert_num_pixels_across_each_expected_cropping_window}

    _pre_serialization_funcs_ = \
        {**_cls_alias._pre_serialization_funcs_,
         "num_pixels_across_each_cbed_pattern": \
         _pre_serialize_num_pixels_across_each_cbed_pattern,
         "num_pixels_across_each_expected_cropping_window": \
         _pre_serialize_num_pixels_across_each_expected_cropping_window}

    _de_pre_serialization_funcs_ = \
        {**_cls_alias._de_pre_serialization_funcs_,
         "num_pixels_across_each_cbed_pattern": \
         _de_pre_serialize_num_pixels_across_each_cbed_pattern,
         "num_pixels_across_each_expected_cropping_window": \
         _de_pre_serialize_num_pixels_across_each_expected_cropping_window}



    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 sampling_grid_dims_in_pixels,
                 least_squares_alg_params,
                 device_name,
                 num_pixels_across_each_expected_cropping_window,
                 skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}

        module_alias = emicroml.modelling.cbed._common
        cls_alias = module_alias._DefaultCBEDPatternGenerator
        cls_alias.__init__(self, ctor_params)

        return None



    def execute_post_core_attrs_update_actions(self):
        super().execute_post_core_attrs_update_actions()

        self._set_fixed_attrs()

        return None



    def _set_fixed_attrs(self):
        self._min_num_non_clipped_disks_in_any_cbed_pattern = 1
        self._principal_disk_idx = 0

        return None



    def _generate_mask_frame(self, distortion_model):
        mask_frame = tuple(0*elem
                           for elem
                           in super()._generate_mask_frame(distortion_model))

        return mask_frame



    def _generate_e_support(self):
        e_support = abs(rng.uniform(low=0.0, high=0.4))

        return e_support



    def _generate_undistorted_disk_support_centers(
            self,
            u_a_support,
            undistorted_tds_model_1,
            undistorted_outer_illumination_shape,
            distortion_model):
        kwargs = \
            {"u_a_support": \
             u_a_support,
             "undistorted_outer_illumination_shape": \
             undistorted_outer_illumination_shape,
             "distortion_model": \
             distortion_model}
        undistorted_principal_disk_support_center = \
            self._generate_undistorted_principal_disk_support_center(**kwargs)

        kwargs["undistorted_principal_disk_support_center"] = \
            undistorted_principal_disk_support_center
        undistorted_satellite_disk_support_centers = \
            self._generate_undistorted_satellite_disk_support_centers(**kwargs)

        insertion_idx = self._principal_disk_idx

        undistorted_disk_support_centers = \
            (undistorted_satellite_disk_support_centers[:insertion_idx]
             + (undistorted_principal_disk_support_center,)
             + undistorted_satellite_disk_support_centers[insertion_idx+1:])

        return undistorted_disk_support_centers



    def _generate_undistorted_principal_disk_support_center(
            self,
            u_a_support,
            undistorted_outer_illumination_shape,
            distortion_model):
        undistorted_outer_illumination_shape_core_attrs = \
            undistorted_outer_illumination_shape.get_core_attrs(deep_copy=False)

        key = \
            ("center"
             if ("center" in undistorted_outer_illumination_shape_core_attrs)
             else "radial_reference_pt")
        u_x_c_OI, u_y_c_OI = \
            undistorted_outer_illumination_shape_core_attrs[key]

        undistorted_OI_params = undistorted_outer_illumination_shape_core_attrs

        kwargs = {"low": 0, "high": 2*np.pi}
        u_theta_OI = self._rng.uniform(**kwargs)

        if "center" in undistorted_OI_params:
            u_x_c_OI, u_y_c_OI = undistorted_OI_params["center"]
            a_OI = undistorted_OI_params["semi_major_axis"]
            e_OI = undistorted_OI_params["eccentricity"]
            theta_OI = undistorted_OI_params["rotation_angle"]

            R_OI = a_OI * np.sqrt((1 - e**2)
                                  / (1 - (e*np.cos(u_theta_OI+theta_OI))**2))
        else:
            u_x_c_OI, u_y_c_OI = undistorted_OI_params["radial_reference_pt"]
            D_OI = undistorted_OI_params["radial_amplitudes"]
            phi_OI = undistorted_OI_params["radial_phases"]
            N_phi_OI = len(phi_OI)
            
            R_OI = D_OI[0]
            for n in range(1, N_phi_OI+1):
                R_OI += D_OI[n]*np.cos(n*u_theta_OI - phi_OI[n-1])

        kwargs = {"low": 0, "high": max(0, R_OI-1.2*u_a_support)}
        u_r_OI = self._rng.uniform(**kwargs)

        undistorted_principal_disk_support_center = \
            ((u_x_c_OI + u_r_OI*np.cos(u_theta_OI)).item(),
             (u_y_c_OI + u_r_OI*np.sin(u_theta_OI)).item())

        return undistorted_principal_disk_support_center



    def _generate_undistorted_satellite_disk_support_centers(
            self, u_a_support, undistorted_principal_disk_support_center):
        rng = self._rng

        kwargs = {"low": self._min_num_non_clipped_disks_in_any_cbed_pattern-1,
                  "high": self._max_num_disks_in_any_cbed_pattern}
        target_num_satellite_disks = self._rng.integers(**kwargs).item()

        undistorted_satellite_disk_support_centers = tuple()
        for _ in range(target_num_satellite_disks):
            u_x_c_0, u_y_c_0 = undistorted_principal_disk_support_center

            min_distance_from_principal_disk_support_center = \
                (u_a_support
                 + (4/self._num_pixels_across_each_cbed_pattern))
            max_distance_from_principal_disk_support_center = \
                ((0.5
                  * self._num_pixels_across_each_expected_cropping_window
                  / self._num_pixels_across_each_cbed_pattern)
                 + ((3/4)*u_a_support))

            kwargs = {"low": min_distance_from_principal_disk_support_center,
                      "high": max_distance_from_principal_disk_support_center}
            distance_from_principal_disk_support_center = rng.uniform(**kwargs)
            d = distance_from_principal_disk_support_center

            kwargs = {"low": 0, "high": 2*np.pi}
            phi = rng.uniform(**kwargs)

            undistorted_satellite_disk_support_center = \
                ((u_x_c_0 + d*np.cos(phi)).item(),
                 (u_y_c_0 + d*np.sin(phi)).item())
            
            undistorted_satellite_disk_support_centers += \
                (undistorted_satellite_disk_support_center,)

        return undistorted_satellite_disk_support_centers



def _check_and_convert_num_pixels_across_each_cropping_window(params):
    obj_name = "num_pixels_across_each_cropping_window"
    
    func_alias = czekitout.convert.to_positive_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    num_pixels_across_each_cropping_window = func_alias(**kwargs)

    divisor = _generate_divisor_2()

    current_func_name = ("_check_and_convert"
                         "_num_pixels_across_each_cropping_window")

    if num_pixels_across_each_cropping_window % divisor != 0:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(divisor)
        raise ValueError(err_msg)

    return num_pixels_across_each_cropping_window



def _pre_serialize_num_pixels_across_each_cropping_window(
        num_pixels_across_each_cropping_window):
    obj_to_pre_serialize = num_pixels_across_each_cropping_window
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_num_pixels_across_each_cropping_window(
        serializable_rep):
    num_pixels_across_each_cropping_window = serializable_rep

    return num_pixels_across_each_cropping_window



_cls_alias = fancytypes.PreSerializableAndUpdatable
class _DefaultCroppedCBEDPatternGenerator(_cls_alias):
    ctor_param_names = ("num_pixels_across_each_cbed_pattern",
                        "max_num_disks_in_any_cbed_pattern",
                        "rng_seed",
                        "sampling_grid_dims_in_pixels",
                        "least_squares_alg_params",
                        "device_name",
                        "num_pixels_across_each_cropping_window")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}

    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs



    def __init__(self, ctor_params):
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        self._rng = np.random.default_rng(self._rng_seed)
        self._device = _get_device(device_name=self._device_name)

        kwargs = {**self_core_attrs,
                  "num_pixels_across_each_expected_cropping_window": \
                  self._num_pixels_across_each_cropping_window}
        del kwargs["disk_boundary_sample_size"]
        del kwargs["num_pixels_across_each_cropping_window"]
        self._cbed_pattern_generator = _DefaultCBEDPatternGenerator(**kwargs)

        j = np.log2(num_pixels_across_each_cropping_window).item()
        j = (np.round(j) if (np.isclose(j, round(j))) else np.ceil(j)).item()
        self._disk_boundary_sample_size = round(4*4*2**j)

        self._set_fixed_attrs()

        return None



    def _set_fixed_attrs(self):
        self._principal_disk_idx = \
            self._cbed_pattern_generator._principal_disk_idx
        
        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def generate(self):
        r"""Generate a cropped fake CBED pattern.

        The randomization scheme employed by the current class to generate
        random cropped fake CBED patterns is somewhat convoluted, and will not
        be documented here in detail. For those who are interested, you can
        parse through the source code of the current class for more details on
        the scheme.

        Returns
        -------
        cropped_cbed_pattern : :class:`fakecbed.discretized.CroppedCBEDPattern`
            The cropped fake CBED pattern.

        """
        generation_attempt_count = 0
        max_num_generation_attempts = 10
        cropped_cbed_pattern_generation_has_not_been_completed = True
        
        while cropped_cbed_pattern_generation_has_not_been_completed:
            try:
                cropped_cbed_pattern_params = \
                    self._generate_cropped_cbed_pattern_params()
                kwargs = \
                    cbed_pattern_params
                cropped_cbed_pattern = \
                    fakecbed.discretized.CroppedCBEDPattern(**kwargs)

                principal_disk_is_overlapping = \
                    cropped_cbed_pattern.principal_disk_is_overlapping
                principal_disk_is_clipped = \
                    cropped_cbed_pattern.principal_disk_is_clipped
                
                if principal_disk_is_overlapping or principal_disk_is_clipped:
                    err_msg = _default_cropped_cbed_pattern_generator_err_msg_1
                    raise ValueError(err_msg)

                mask_frame = self._generate_mask_frame(cropped_cbed_pattern)

                kwargs = {"new_core_attr_subset_candidate": \
                          {"mask_frame": mask_frame},
                          "skip_validation_and_conversion": \
                          True}
                cropped_cbed_pattern.update(**kwargs)
                
                cropped_cbed_pattern.get_signal(deep_copy=False)

                cropped_cbed_pattern_generation_has_not_been_completed = False
            except:
                generation_attempt_count += 1
                
                if generation_attempt_count == max_num_generation_attempts:
                    unformatted_err_msg = \
                        _default_cropped_cbed_pattern_generator_err_msg_2

                    args = ("", " ({})".format(max_num_generation_attempts))
                    err_msg = unformatted_err_msg.format(*args)
                    raise RuntimeError(err_msg)

        return cropped_cbed_pattern



    def _generate_cropped_cbed_pattern_params(self):
        cbed_pattern = \
            self._cbed_pattern_generator.generate()
        cropping_window_center = \
            self._generate_cropping_window_center(cbed_pattern)
        cropping_window_dims_in_pixels = \
            2*(self._num_pixels_across_each_cropping_window,)

        cropped_cbed_pattern_params = \
            {"cbed_pattern": cbed_pattern,
             "cropping_window_center": cropping_window_center,
             "cropping_window_dims_in_pixels": cropping_window_dims_in_pixels,
             "principal_disk_idx": 0,
             "disk_boundary_sample_size": self._disk_boundary_sample_size,
             "mask_frame": 4*(0,)}

        return cropped_cbed_pattern_params



    def _generate_cropping_window_center(self, cbed_pattern):
        q_x_c_and_q_y_c = \
            self._generate_q_x_c_and_q_y_c_of_principal_disk(**kwargs)

        disk_supports = cbed_pattern.get_disk_supports(deep_copy=False)
        q_x, q_y = self._generate_q_x_and_q_y_of_cbed_pattern_signal()

        disk_support_COMs_shape = (cbed_pattern.num_disks, 2)
        disk_support_COMs = torch.zeros(disk_support_COMs_shape,
                                        device=self._device)

        disk_support_COMs[:, 0] = \
            ((q_x[None, :, :]*disk_supports).sum(dim=(1, 2))
             / disk_supports.sum(dim=(1, 2)))
        disk_support_COMs[:, 1] = \
            ((q_y[None, :, :]*disk_supports).sum(dim=(1, 2))
             / disk_supports.sum(dim=(1, 2)))

        principal_disk_idx = self._principal_disk_idx

        displacements = (disk_support_COMs
                         - disk_support_COMs[principal_disk_idx])
        distances = torch.linalg.norm(displacements, dim=-1)
        nn_distance = torch.sort(distances).values[1].item()

        kwargs = {"low": 0, "high": 2*np.pi}
        phi = self._rng.uniform(**kwargs)
        
        kwargs = {"low": 0, "high": 0.4*nn_distance}
        R = self._rng.uniform(**kwargs)

        cropping_window_center = (q_x_c + R*np.cos(phi).item(),
                                  q_y_c + R*np.sin(phi).item())
        
        return cropping_window_center



    def _generate_q_x_c_and_q_y_c_of_principal_disk(self, cbed_pattern):
        cbed_pattern_core_attrs = cbed_pattern.get_core_attrs(deep_copy=False)
        
        undistorted_disks = cbed_pattern_core_attrs["undistorted_disks"]
        principal_disk_idx = self._principal_disk_idx
        undistorted_disk = undistorted_disks[principal_disk_idx]

        distortion_model = cbed_pattern_core_attrs["distortion_model"]

        distortion_model_core_attrs = \
            distortion_model.get_core_attrs(deep_copy=False)
        coord_transform_params = \
            distortion_model_core_attrs["coord_transform_params"]
        
        undistorted_disk_core_attrs = \
            undistorted_disk.get_core_attrs(deep_copy=False)
        u_x_c, u_y_c = \
            undistorted_disk_core_attrs_core_attrs["center"]

        kwargs = {"u_x": torch.tensor(((u_x_c,),), device=self._device),
                  "u_y": torch.tensor(((u_y_c,),), device=self._device),
                  "coord_transform_params": coord_transform_params,
                  "device": self._device,
                  "skip_validation_and_conversion": True}
        q_x, q_y = distoptica.apply_coord_transform(**kwargs)

        q_x_c_and_q_y_c_of_principal_disk = (q_x[0, 0].item(), q_y[0, 0].item())

        return q_x_c_and_q_y_c_of_principal_disk



    def _generate_q_x_and_q_y_of_cbed_pattern_signal(self):
        size = self._num_pixels_across_each_cbed_pattern
        scale = 1/size
        offset = 0.5*scale

        pair_of_1d_coord_arrays = \
            (scale*torch.arange(size, device=self._device)+offset,
             1 - (scale*torch.arange(size, device=self._device)+offset))

        generate_q_x_and_q_y_of_cbed_pattern_signal = \
            torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")

        return generate_q_x_and_q_y_of_cbed_pattern_signal



    def _generate_mask_frame(self, cropped_cbed_pattern):
        get_fully_qualified_class_name = \
            czekitout.name.fully_qualified_class_name
        ml_model_task = \
            "/".join(get_fully_qualified_class_name(self).split(".")[2:-1])

        num_pixels_across_each_cropping_window = \
            self._num_pixels_across_each_cropping_window
        
        d_q = 1/num_pixels_across_each_cropping_window

        method_name = ("get_principal_disk_bounding_box_"
                       "in_cropped_image_fractional_coords")
        method_alias = getattr(cropped_cbed_pattern, method_name)
        bounding_box = method_alias(deep_copy=False)

        if ml_model_task == "cbed/disk/localization":
            bounding_box_buffer = 4*d_q*np.ones((4,))
        else:
            kwargs = {"low": 4*d_q,"high": max(1/10, 4*d_q), "size": 4}
            bounding_box_buffer = self._rng.uniform(**kwargs)

        quadruple_1 = np.array(max(bounding_box[0]-bounding_box_buffer[0], 0),
                               max(1-bounding_box[1]-bounding_box_buffer[1], 0),
                               max(bounding_box[2]-bounding_box_buffer[2], 0),
                               max(1-bounding_box[3]-bounding_box_buffer[3], 0))

        if ml_model_task == "cbed/disk/localization":
            kwargs = {"low": 0/4, "high": 1/4, "size": 4}
            quadruple_2 = self._rng.uniform(**kwargs)
            p = (1/2, 1-1/2)
        else:
            quadruple_2 = quadruple_1
            p = (1, 0)

        kwargs = \
            {"a": (True, False), "p": p}
        trivial_mask_frame_is_not_to_be_generated = \
            self._rng.choice(**kwargs).item()

        mask_frame = \
            tuple(np.round(((quadruple_1<quadruple_2)*quadruple_1
                            + (quadruple_1>=quadruple_2)*quadruple_2)
                           * num_pixels_across_each_cropping_window).astype(int)
                  * trivial_mask_frame_is_not_to_be_generated)

        return mask_frame



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
        ("cropped_cbed_pattern_images",
         "cropped_disk_overlap_maps",
         "cropped_principal_disk_supports")

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
        _generate_keys_related_to_mra()
    keys_of_normalizable_ml_data_dict_elems_not_having_decoders += \
        ("principal_disk_bounding_boxes", "principal_disk_boundary_pt_sets")

    return keys_of_normalizable_ml_data_dict_elems_not_having_decoders



def _generate_keys_related_to_mra():
    accepted_wavelet_names = _generate_accepted_wavelet_names()

    keys_related_to_mra = tuple()
    for wavelet_name in accepted_wavelet_names:
        unformatted_key = ("max_level_{}_approx_coeff_sets"
                           "_of_principal_disk_boundary_pt_sets")
        keys_related_to_mra += (unformatted_key.format(wavelet_name),)

    return keys_related_to_mra



def _generate_accepted_wavelet_names():
    accepted_wavelet_family_name = _generate_accepted_wavelet_family_name()
    accepted_wavelet_names = tuple(accepted_wavelet_family_name+str(idx)
                                   for idx in (1, 2, 4, 8, 16, 32))

    return accepted_wavelet_names



def _generate_accepted_wavelet_family_name():
    accepted_wavelet_family_name = "db"

    return accepted_wavelet_family_name



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



def _generate_keys_of_ml_data_dict_elems_not_having_decoders():
    unformatted_func_name = ("_generate_keys_of_{}normalizable"
                             "_ml_data_dict_elems_not_having_decoders")
    func_names = (unformatted_func_name.format(""),
                  unformatted_func_name.format("un"))
    global_symbol_table = globals()

    keys_of_ml_data_dict_elems_not_having_decoders = tuple()
    for func_name in func_names:
        func_alias = global_symbol_table[func_name]
        keys_of_ml_data_dict_elems_not_having_decoders += func_alias()

    return keys_of_ml_data_dict_elems_not_having_decoders



def _generate_cropped_cbed_pattern_signal(cropped_cbed_pattern_generator):
    current_func_name = "_generate_cropped_cbed_pattern_signal"

    try:
        cropped_cbed_pattern = cropped_cbed_pattern_generator.generate()
    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        args = (" ``cropped_cbed_pattern_generator``", "")
        err_msg = unformatted_err_msg.format(*args)
        raise RuntimeError(err_msg)

    try:
        accepted_types = (fakecbed.discretized.CroppedCBEDPattern,)
        kwargs = {"obj": cropped_cbed_pattern,
                  "obj_name": "cropped_cbed_pattern",
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = globals()[current_func_name+"_err_msg_2"]
        raise TypeError(err_msg)

    cropped_cbed_pattern_signal = \
        cropped_cbed_pattern.get_signal(deep_copy=False)

    return cropped_cbed_pattern_signal



def _check_cropped_cbed_pattern_signal(cropped_cbed_pattern_signal):
    principal_disk_is_overlapping = \
        cropped_cbed_pattern.principal_disk_is_overlapping
    principal_disk_is_clipped = \
        cropped_cbed_pattern.principal_disk_is_clipped
    cropped_cbed_pattern_dims_in_pixels = \
        cropped_cbed_pattern_signal.axes_manager.signal_shape
                
    current_func_name = "_check_cropped_cbed_pattern_signal"
    
    if principal_disk_is_overlapping or principal_disk_is_clipped:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    dims_in_pixels = cropped_cbed_pattern_dims_in_pixels
    if (dims_in_pixels[0] != dims_in_pixels[1]):
        err_msg = globals()[current_func_name+"_err_msg_2"]
        raise ValueError(err_msg)

    try:
        params = {"num_pixels_across_each_cropping_window": \
                  cropped_cbed_pattern_dims_in_pixels}
        _ = _check_and_convert_num_pixels_across_each_cropping_window(params)
    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_3"]
        err_msg = unformatted_err_msg.format(divisor)
        raise ValueError(err_msg)
    
    return None



def _extract_ml_data_dict_from_cropped_cbed_pattern_signal(
        cropped_cbed_pattern_signal):
    path_to_item = \
        ("FakeCBED"
         ".principal_disk_boundary_pts_in_cropped_image_fractional_coords")
    principal_disk_boundary_pt_set = \
        np.array(cropped_cbed_pattern_signal.metadata.get_item(path_to_item))

    path_to_item = \
        ("FakeCBED"
         ".principal_disk_bounding_box_in_cropped_image_fractional_coords")
    principal_disk_bounding_box = \
        np.array(cropped_cbed_pattern_signal.metadata.get_item(path_to_item))

    path_to_item = \
        ("FakeCBED.pre_serialized_core_attrs.principal_disk_idx")
    principal_disk_idx = \
        cropped_cbed_pattern_signal.metadata.get_item(path_to_item)

    keys_related_to_mra = _generate_keys_related_to_mra()

    ml_data_dict = {"cropped_cbed_pattern_images": \
                    cropped_cbed_pattern_signal.data[0],
                    "cropped_disk_overlap_maps": \
                    cropped_cbed_pattern_signal.data[2],
                    "cropped_principal_disk_supports": \
                    cropped_cbed_pattern_signal.data[3+principal_disk_idx],
                    "principal_disk_boundary_pt_sets": \
                    principal_disk_boundary_pt_set,
                    "principal_disk_bounding_boxes": \
                    principal_disk_bounding_box}
    for key in keys_related_to_mra:
        wavelet_name = key.split("_")[2]
        j_vdash = _get_j_vdash_from_wavelet_name(wavelet_name)
        ml_data_dict[key] = np.zeros((2**j_vdash, 2))
    for key in ml_data_dict:
        ml_data_dict[key] = np.expand_dim(ml_data_dict[key], axis=0)

    # For each key ``key`` in
    # ``keys_related_to_mra+("principal_disk_boundary_pt_sets",)``,
    # ``ml_data_dict[key]`` is updated via the function
    # ``_update_ml_dataset_output_file``, which is called near the end of the
    # functions ``_generate_and_save_ml_dataset`` and
    # ``_combine_ml_dataset_files``.

    return ml_data_dict



_module_alias = emicroml.modelling.cbed._common
_tol_for_comparing_floats = _module_alias._tol_for_comparing_floats



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._UnnormalizedMLDataInstanceGenerator
class _UnnormalizedMLDataInstanceGenerator(_cls_alias):
    def __init__(self, cropped_cbed_pattern_generator):
        self._cropped_cbed_pattern_generator = cropped_cbed_pattern_generator

        self._expected_cropped_cbed_pattern_dims_in_pixels = None

        kwargs = \
            {"cropped_cbed_pattern_generator": cropped_cbed_pattern_generator}
        cropped_cbed_pattern_signal = \
            _generate_cropped_cbed_pattern_signal(**kwargs)
        
        self._expected_cropped_cbed_pattern_dims_in_pixels = \
            cropped_cbed_pattern_signal.axes_manager.signal_shape

        path_to_item = \
            ("FakeCBED.pre_serialized_core_attrs.disk_boundary_sample_size")
        self._expected_disk_boundary_sample_size = \
            cropped_cbed_pattern_signal.metadata.get_item(path_to_item)

        kwargs = {"cropped_cbed_pattern_signal": cropped_cbed_pattern_signal}
        _check_cropped_cbed_pattern_signal(**kwargs)

        cached_ml_data_instances = self._generate(num_ml_data_instances=1)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._UnnormalizedMLDataInstanceGenerator
        kwargs = {"cached_ml_data_instances": cached_ml_data_instances}
        cls_alias.__init__(self, **kwargs)

        return None



    def _generate_ml_data_dict_containing_only_one_ml_data_instance(self):
        cropped_cbed_pattern_generator = \
            self._cropped_cbed_pattern_generator

        kwargs = \
            {"cropped_cbed_pattern_generator": cropped_cbed_pattern_generator}
        cropped_cbed_pattern_signal = \
            _generate_cropped_cbed_pattern_signal(**kwargs)

        kwargs = {"cropped_cbed_pattern_signal": cropped_cbed_pattern_signal}
        _check_cropped_cbed_pattern_signal(**kwargs)

        cropped_cbed_pattern_dims_in_pixels = \
            cropped_cbed_pattern_signal.axes_manager.signal_shape
        
        expected_cropped_cbed_pattern_dims_in_pixels = \
            self._expected_cropped_cbed_pattern_dims_in_pixels

        if (cropped_cbed_pattern_dims_in_pixels
            != expected_cropped_cbed_pattern_dims_in_pixels):
            err_msg = _unnormalized_ml_data_instance_generator_err_msg_1
            raise ValueError(err_msg)

        path_to_item = \
            ("FakeCBED.pre_serialized_core_attrs.disk_boundary_sample_size")
        disk_boundary_sample_size = \
            cropped_cbed_pattern_signal.metadata.get_item(path_to_item)

        expected_disk_boundary_sample_size = \
            self._expected_disk_boundary_sample_size = \

        if (disk_boundary_sample_size
            != expected_disk_boundary_sample_size):
            err_msg = _unnormalized_ml_data_instance_generator_err_msg_2
            raise ValueError(err_msg)

        method_alias = \
            super()._generate_ml_data_dict_containing_only_one_ml_data_instance
        ml_data_dict = \
            method_alias()

        func_alias = _extract_ml_data_dict_from_cropped_cbed_pattern_signal
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
        if "cropped" in key:
            shape_template = (variable_axis_size_dict_keys[0],
                              variable_axis_size_dict_keys[1],
                              variable_axis_size_dict_keys[1])
        elif "box" in key:
            shape_template = (variable_axis_size_dict_keys[0],
                              4)
        elif key == "principal_disk_boundary_pt_sets":
            shape_template = (variable_axis_size_dict_keys[0],
                              variable_axis_size_dict_keys[2],
                              2)
        else:
            wavelet_name = key.split("_")[2]
            j_vdash = _get_j_vdash_from_wavelet_name(wavelet_name)
            shape_template = (variable_axis_size_dict_keys[0], 2**j_vdash, 2)

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



def _get_j_vdash_from_wavelet_name(wavelet_name):
    wavelet = pywt.Wavelet(wavelet_name)
    j_vdash = _get_j_vdash_from_wavelet(wavelet)

    return j_vdash



def _get_j_vdash_from_wavelet(wavelet):
    beta = _get_scaling_function_support_of_wavelet(wavelet)

    j = np.log2(beta)
    j = (np.round(j) if (np.isclose(j, round(j))) else np.ceil(j)).item()
    j = round(j)

    j_vdash = j

    return j_vdash



def _get_scaling_function_support_of_wavelet(wavelet):
    scaling_function_support_of_wavelet = len(wavelet.dec_lo)-1

    return scaling_function_support_of_wavelet



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



def _calc_j_dashv_candidate(principal_disk_boundary_pt_set,
                            cropped_cbed_pattern_dims_in_pixels):
    kwargs = {"boundary_pt_set": principal_disk_boundary_pt_set}
    L = _calc_approx_boundary_curve_length(**kwargs)

    j_dashv_candidate = _calc_j_dashv_candidate_lower_limit()

    j = np.log2(cropped_cbed_pattern_dims_in_pixels[0]*L)
    j = (np.round(j) if (np.isclose(j, round(j))) else np.ceil(j)).item()
    j = round(j)

    j_dashv_candidate = max(j_dashv_candidate, j)
            
    return j_dashv_candidate



def _calc_j_dashv_candidate_lower_limit():
    wavelet_names = _generate_accepted_wavelet_names()
    wavelets = tuple(pywt.Wavelet(wavelet_name)
                     for wavelet_name in wavelet_names)

    j_dashv_candidate_lower_limit = 0

    for wavelet in wavelets:
        j_vdash = _get_j_vdash_from_wavelet(wavelet)
        j_dashv_candidate_lower_limit = max(j_dashv_candidate_lower_limit,
                                            j_vdash+1)

    return j_dashv_candidate_lower_limit



def _generate_accepted_wavelets():
    accepted_wavelet_names = _generate_accepted_wavelet_names()
    accepted_wavelet = tuple(pywt.Wavelet(wavelet_name)
                             for wavelet_name in accepted_wavelet_names)

    return accepted_wavelet



def _calc_approx_boundary_curve_length(boundary_pt_set):
    s = _arc_lengths_corresponding_to_boundary_pt_set(boundary_pt_set)
    approx_boundary_curve_length = s[-1]

    return approx_boundary_curve_length



def _arc_lengths_corresponding_to_boundary_pt_set(boundary_pt_set):
    kwargs = {"a": boundary_pt_set, "axis": 0, "append": boundary_pt_set[:1]}
    displacements = np.diff(**kwargs)

    kwargs = {"x": displacements, "axis": 1}
    distances = np.linalg.norm(**kwargs)

    s = (0,) + tuple(np.cumsum(distances).tolist())

    return s



def _reinterpolate_boundary_pt_sets(interpolated_boundary_pt_sets,
                                    num_pts_in_reinterpolation):
    num_sets = interpolated_boundary_pt_sets.shape[0]
    num_cartesian_cmpnts = interpolated_boundary_pt_sets.shape[2]

    kwargs = {"shape": \
              (num_sets, num_pts_in_reinterpolation, num_cartesian_cmpnts),
              "dtype": \
              interpolated_boundary_pt_sets.dtype}
    reinterpolated_boundary_pt_sets = np.zeros(**kwargs)

    for set_idx in range(num_sets):
        kwargs = {"interpolated_boundary_pt_set": \
                  interpolated_boundary_pt_sets[set_idx],
                  "num_pts_in_reinterpolation": \
                  num_pts_in_reinterpolation}
        func_alias = _reinterpolate_boundary_pt_set
        reinterpolated_boundary_pt_sets[set_idx] = func_alias(**kwargs)

    return reinterpolated_boundary_pt_sets



def _reinterpolate_boundary_pt_set(interpolated_boundary_pt_set,
                                   num_pts_in_reinterpolation):
    num_cartesian_cmpnts = interpolated_boundary_pt_set.shape[1]

    kwargs = {"boundary_pt_set": interpolated_boundary_pt_set}
    s = _arc_lengths_corresponding_to_boundary_pt_set(**kwargs)

    max_horizontal_coord_idx = np.argmax(interpolated_boundary_pt_set[:, 0])

    kwargs = {"shape": (num_pts_in_reinterpolation, num_cartesian_cmpnts),
              "dtype": interpolated_boundary_pt_set.dtype}
    reinterpolated_boundary_pt_set = np.zeros(**kwargs)

    data_on_which_to_eval_interpolant = (np.arange(num_pts_in_reinterpolation)
                                         / num_pts_in_reinterpolation)

    kwargs = {"a": s, "shift": -max_horizontal_coord_idx}
    rolled_s = np.roll(**kwargs)

    independent_data = ((rolled_s-s[max_horizontal_coord_idx])/s[-1])%1
    independent_data = np.append(independent_data, 1)

    for cartesian_cmpnt_idx in range(num_cartesian_cmpnts):
        kwargs = {"a": interpolated_boundary_pt_set[:, cartesian_cmpnt_idx],
                  "shift": -max_horizontal_coord_idx}
        rolled_interpolated_boundary_pt_set = np.roll(**kwargs)

        dependent_data = rolled_interpolated_boundary_pt_set
        dependent_data = np.append(dependent_data, dependent_data[0])

        kwargs = \
            {"x": data_on_which_to_eval_interpolant,
             "xp": independent_data,
             "yp": dependent_data}
        reinterpolated_boundary_pt_set[:, cartesian_cmpnt_idx] = \
            np.interp(**kwargs)

    return reinterpolated_boundary_pt_set



def _perform_mra_of_principal_disk_boundary_pt_sets(
        principal_disk_boundary_pt_sets):
    num_sets = principal_disk_boundary_pt_sets.shape[0]
    j_dashv = round(np.log2(principal_disk_boundary_pt_sets.shape[1]))
    num_cartesian_cmpnts = principal_disk_boundary_pt_sets.shape[2]

    mra_results_of_principal_disk_boundary_pt_sets = \
        {"principal_disk_boundary_pt_sets": principal_disk_boundary_pt_sets}

    key_subset = _generate_keys_related_to_mra()

    for key in key_subset:
        wavelet_name = key.split("_")[2]
        wavelet = pywt.Wavelet(wavelet_name)
        j_vdash = _get_j_vdash_from_wavelet(wavelet)

        kwargs = {"shape": (num_sets, 2**j_vdash, num_cartesian_cmpnts),
                  "dtype": principal_disk_boundary_pt_sets.dtype}
        subset_of_mra_results = np.zeros(**kwargs)

        for set_idx in range(num_sets):
            principal_disk_boundary_pt_set = \
                principal_disk_boundary_pt_sets[set_idx]

            for cartesian_cmpnt_idx in range(num_cartesian_cmpnts):
                kwargs = \
                    {"data": \
                     principal_disk_boundary_pt_set[cartesian_cmpnt_idx],
                     "wavelet": \
                     wavelet,
                     "mode": \
                     "periodization",
                     "level": \
                     j_dashv-j_vdash}
                subset_of_mra_results[set_idx, :, cartesian_cmpnt_idx] = \
                    pywt.wavedec(**kwargs)[0]

        mra_results_of_principal_disk_boundary_pt_sets[key] = \
            subset_of_mra_results

    return mra_results_of_principal_disk_boundary_pt_sets



def _save_data_chunk(starting_idx_offset,
                     chunk_idx,
                     max_num_ml_data_instances_per_chunk,
                     data_chunk,
                     output_hdf5_dataset):
    kwargs = locals()
    module_alias = emicroml.modelling._common
    func_alias = module_alias._save_data_chunk
    func_alias(**kwargs)

    return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataNormalizer
class _MLDataNormalizer(_cls_alias):
    def __init__(self, max_num_ml_data_instances_per_file_update):
        self._ml_data_dict_key_subset_1 = \
            ("principal_disk_boundary_pt_sets",)
        self._ml_data_dict_key_subset_2 = \
            _generate_keys_related_to_mra()
        self._ml_data_dict_key_subset_3 = \
            self._ml_data_dict_key_subset_1 + self._ml_data_dict_key_subset_2

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



    def _update_extrema_cache(self, ml_data_dict):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        super()._update_extrema_cache(**kwargs)

        self._j_dashv = self._calc_j_dashv(**kwargs)
        self._N_dot = 2**self._j_dashv

        return None



    def _calc_j_dashv(self, ml_data_dict):
        key = "cropped_cbed_pattern_images"
        cropped_cbed_pattern_dims_in_pixels = ml_data_dict[key].shape[1:]

        key = "principal_disk_boundary_pt_sets"
        principal_disk_boundary_pt_sets = ml_data_dict[key]

        j_dashv = 0

        for principal_disk_boundary_pt_set in principal_disk_boundary_pt_sets:
            kwargs = {"principal_disk_boundary_pt_set": \
                      principal_disk_boundary_pt_set,
                      "cropped_cbed_pattern_dims_in_pixels": \
                      cropped_cbed_pattern_dims_in_pixels}
            j_dashv_candidate = _calc_j_dashv_candidate(**kwargs)

            j_dashv = max(j_dashv, j_dashv_candidate)

        return j_dashv



    def _normalize_ml_dataset_file(self, path_to_ml_dataset, print_msgs):
        kwargs = {"path_to_ml_dataset": path_to_ml_dataset}
        self._perform_mra_of_ml_dataset_file(**kwargs)

        kwargs["print_msgs"] = print_msgs
        super()._normalize_ml_dataset_file(**kwargs)

        return None



    def _perform_mra_of_ml_dataset_file_and_save_to_said_file(
            self, path_to_ml_dataset):
        kwargs = {"path_to_ml_dataset": path_to_ml_dataset}
        method_name = ("_resize_and_or_copy_hdf5_datasets"
                       "_storing_pts_of_ml_dataset_file")
        method_alias = getattr(self, method_name)
        method_alias(**kwargs)

        kwargs = {"path_to_ml_dataset": \
                  path_to_ml_dataset,
                  "hdf5_dataset_path": \
                  self._ml_data_dict_key_subset_1[0] + "_copy"}
        hdf5_dataset = self._get_hdf5_dataset(**kwargs)

        kwargs = \
            {"hdf5_dataset": hdf5_dataset}
        max_num_ml_data_instances_per_chunk = \
            self._calc_max_num_ml_data_instances_per_chunk(**kwargs)
        num_chunks = \
            self._calc_num_chunks(**kwargs)

        for chunk_idx in range(num_chunks):
            method_name = ("_perform_mra_of_ml_data_instance_chunk"
                           "_and_save_to_output_file")
            method_alias = getattr(self, method_name)
            kwargs = {"chunk_idx": \
                      chunk_idx,
                      "max_num_ml_data_instances_per_chunk": \
                      max_num_ml_data_instances_per_chunk,
                      "hdf5_dataset": \
                      hdf5_dataset}
            method_alias(**kwargs)

        file_obj = hdf5_dataset.file
        
        for key in self._ml_data_dict_key_subset_3:
            hdf5_dataset_path = key
            self._extrema_cache[key]["min"] = file_obj[hdf5_dataset_path].min()
            self._extrema_cache[key]["max"] = file_obj[hdf5_dataset_path].max()
        del file_obj[hdf5_dataset.name]

        self._update_normalization_weights_and_biases()
        
        file_obj.close()

        return None



    def _resize_and_or_copy_hdf5_datasets_storing_pts_of_ml_dataset_file(
            self, path_to_ml_dataset):
        hdf5_dataset_A_path = self._ml_data_dict_key_subset_1[0]
        hdf5_dataset_B_path = hdf5_dataset_A_path + "_temp"

        kwargs = {"path_to_ml_dataset": path_to_ml_dataset,
                  "hdf5_dataset_path": hdf5_dataset_A_path}
        hdf5_dataset_A = self._get_hdf5_dataset(**kwargs)

        hdf5_dataset_A_shape = hdf5_dataset_A.shape
        hdf5_dataset_B_shape = (hdf5_dataset_A_shape[0], self._j_dashv, 2)

        file_obj = hdf5_dataset.file

        kwargs = {"name": hdf5_dataset_B_path,
                  "shape": hdf5_dataset_B_shape,
                  "dtype": hdf5_dataset_A.dtype}
        file_obj.create_dataset(**kwargs)
        hdf5_dataset_B = file_obj[hdf5_dataset_B_path]

        for attr_name in hdf5_dataset_A.attrs:
            attr = hdf5_dataset_A.attrs[attr_name]
            hdf5_dataset_B.attrs[attr_name] = attr

        if key in self._ml_data_dict_key_subset_1:
            kwargs = {"source": hdf5_dataset_A,
                      "dest": hdf5_dataset_A_path + "_copy"}
            file_obj.copy(**kwargs)

        del file_obj[hdf5_dataset_A_path]
        file_obj[hdf5_dataset_A_path] = file_obj[hdf5_dataset_B_path]
        del file_obj[hdf5_dataset_B_path]

        file_obj.close()

        return None



    def _perform_mra_of_ml_data_instance_chunk_and_save_to_output_file(
            self,
            chunk_idx,
            max_num_ml_data_instances_per_chunk,
            hdf5_dataset):
        kwargs = \
            {"chunk_idx": \
             chunk_idx,
             "max_num_ml_data_instances_per_chunk": \
             max_num_ml_data_instances_per_chunk,
             "input_hdf5_dataset": \
             hdf5_dataset}
        principal_disk_boundary_pt_sets_from_chunk = \
            self._load_data_chunk(**kwargs)

        kwargs = \
            {"interpolated_boundary_pt_sets": \
             principal_disk_boundary_pt_sets_from_chunk,
             "num_pts_in_reinterpolation": \
             self._N_dot}
        reinterpolated_principal_disk_boundary_pt_sets_from_chunk = \
            _reinterpolate_boundary_pt_sets(**kwargs)

        kwargs = \
            {"principal_disk_boundary_pt_sets": \
             reinterpolated_principal_disk_boundary_pt_sets_from_chunk}
        mra_results_of_principal_disk_boundary_pt_sets_from_chunk = \
            _perform_mra_of_principal_disk_boundary_pt_sets(**kwargs)

        kwargs = {"chunk_idx": \
                  chunk_idx,
                  "unnormalized_output_data_chunks": \
                  mra_results_of_principal_disk_boundary_pt_sets_from_chunk,
                  "file_obj": \
                  hdf5_dataset.file}
        self._save_unnormalized_output_data_chunks(**kwargs)

        return None



    def _save_unnormalized_output_data_chunks(self,
                                              chunk_idx,
                                              unnormalized_output_data_chunks,
                                              file_obj):
        for key in unnormalized_output_data_chunks:
            output_hdf5_path = key
            kwargs = {"starting_idx_offset": \
                      self._ml_data_instance_idx_offset,
                      "chunk_idx": \
                      chunk_idx,
                      "max_num_ml_data_instances_per_chunk": \
                      self._max_num_ml_data_instances_per_chunk,
                      "data_chunk": \
                      unnormalized_output_data_chunks[key],
                      "output_hdf5_dataset": \
                      file_obj[output_hdf5_path]}
            _save_data_chunk(**kwargs)

        return None



    def _normalize_mra_in_ml_dataset_file(self, path_to_ml_dataset):
        copy_of_normalization_weights = self._normalization_weights.copy()
        self._normalization_weights = {key: self._normalization_weights[key]
                                       for key
                                       in self._ml_data_dict_key_subset_3}
        
        kwargs = {"path_to_ml_dataset": path_to_ml_dataset, "print_msgs": False}
        super()._normalize_ml_dataset_file(**kwargs)

        self._normalization_weights = copy_of_normalization_weights

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
        if "support" in key:
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
        if ("image" in key) or ("support" in key):
            unnormalized_value_limits = (0, 1)
        elif "overlap_map" in key:
            unnormalized_value_limits = (0, np.inf)
        else:
            unnormalized_value_limits = (-np.inf, np.inf)
                
        ml_data_dict_key_to_unnormalized_value_limits_map[key] = \
            unnormalized_value_limits

    return ml_data_dict_key_to_unnormalized_value_limits_map



def _generate_ml_data_dict_key_to_custom_value_checker_map():
    ml_data_dict_key_to_custom_value_checker_map = \
        {"cropped_cbed_pattern_images": \
         _custom_value_checker_for_cropped_cbed_pattern_images}

    return ml_data_dict_key_to_custom_value_checker_map



def _custom_value_checker_for_cropped_cbed_pattern_images(
        data_chunk_is_expected_to_be_normalized_if_normalizable,
        key_used_to_get_data_chunk,
        data_chunk,
        name_of_obj_alias_from_which_data_chunk_was_obtained,
        obj_alias_from_which_data_chunk_was_obtained):
    lower_value_limit = 0
    upper_value_limit = 1
    tol = _tol_for_comparing_floats
    current_func_name = "_custom_value_checker_for_cropped_cbed_pattern_images"

    for cropped_cbed_pattern_image in data_chunk:
        image = cropped_cbed_pattern_image

        if ((abs(image.min().item()-lower_value_limit) > tol)
            or (abs(image.max().item()-upper_value_limit) > tol)):
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
        self._ml_data_dict_key_subset_1 = \
            ("principal_disk_boundary_pt_sets",)
        self._ml_data_dict_key_subset_2 = \
            _generate_keys_related_to_mra()
        self._ml_data_dict_key_subset_3 = \
            self._ml_data_dict_key_subset_1 + self._ml_data_dict_key_subset_2
        self._ml_data_dict_key_subset_4 = \
            tuple(set(self._ml_data_dict_keys)
                  - set(self._ml_data_dict_key_subset_3))

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



    def _copy_and_renormalize_all_input_data_and_save_to_output_file(
            self,
            output_ml_dataset_filename,
            rm_input_ml_dataset_files):
        copy_of_ml_data_dict_keys = self._ml_data_dict_keys.copy()
        self._ml_data_dict_keys = self._ml_data_dict_key_subset_4

        copy_and_renormalize_input_data_subset_4_and_save_to_output_file = \
            super()._copy_and_renormalize_all_input_data_and_save_to_output_file

        method_alias = \
            copy_and_renormalize_input_data_subset_4_and_save_to_output_file
        kwargs = \
            {"output_ml_dataset_filename": output_ml_dataset_filename,
             "rm_input_ml_dataset_files": False}
        _ = \
            method_alias(**kwargs)

        self._j_dashv = self._calc_j_dashv(output_ml_dataset_filename)
        self._N_dot = 2**self._j_dashv

        method_alias = \
            self._perform_mra_of_all_input_files_and_save_to_output_file
        kwargs = \
            {"output_ml_dataset_filename": output_ml_dataset_filename}
        _ = \
            method_alias(**kwargs)

        self._normalize_remaining_unnormalized_data_in_output_file(**kwargs)

        self._ml_data_dict_keys = copy_of_ml_data_dict_keys

        if rm_input_ml_dataset_files:
            for input_ml_dataset_filename in self._input_ml_dataset_filenames:
                pathlib.Path(input_ml_dataset_filename).unlink(missing_ok=True)

        return None



    def _calc_j_dashv(self, output_ml_dataset_filename):
        kwargs = {"path_to_ml_dataset": output_ml_dataset_filename,
                  "hdf5_dataset_path": "principal_disk_boundary_pt_sets",
                  "read_only": True}
        output_hdf5_dataset = self._get_hdf5_dataset(**kwargs)

        j_dashv = round(np.log2(output_hdf5_dataset.shape[1]))

        output_hdf5_dataset.file.close()

        return j_dashv



    def _perform_mra_of_all_input_files_and_save_to_output_file(
            self, output_ml_dataset_filename):
        self._ml_data_instance_idx_offset = 0

        for input_ml_dataset_filename in self._input_ml_dataset_filenames:
            method_alias = \
                self._perform_mra_of_input_file_and_save_to_output_file
            kwargs = \
                {"input_ml_dataset_filename": input_ml_dataset_filename,
                 "output_ml_dataset_filename": output_ml_dataset_filename}
            _ = \
                method_alias(**kwargs)

            key = \
                input_ml_dataset_filename
            self._ml_data_instance_idx_offset += \
                self._ml_data_instance_counts_of_input_ml_datasets[key]

        output_file_obj = h5py.File(output_ml_dataset_filename, "r")

        extrema_cache = self._ml_data_normalizer._extrema_cache
        for key in self._ml_data_dict_key_subset_3:
            hdf5_dataset_path = key
            extrema_cache[key]["min"] = output_file_obj[hdf5_dataset_path].min()
            extrema_cache[key]["max"] = output_file_obj[hdf5_dataset_path].max()

        self._ml_data_normalizer._update_normalization_weights_and_biases()

        output_file_obj.close()

        return None



    def _perform_mra_of_input_file_and_save_to_output_file(
            self, input_ml_dataset_filename, output_ml_dataset_filename):
        input_file_obj = h5py.File(input_ml_dataset_filename, "r")
        output_file_obj = h5py.File(output_ml_dataset_filename, "a")

        key = input_ml_dataset_filename
        fraction = (self._ml_data_instance_counts_of_input_ml_datasets[key]
                    / self._max_num_ml_data_instances_per_chunk)
        num_chunks_per_hdf5_dataset = np.ceil(fraction).astype(int)

        hdf5_dataset_path = \
            "principal_disk_boundary_pt_sets"
        reinterpolation_is_required = \
            (input_file_obj[hdf5_dataset_path].shape[1] != self._N_dot)

        for chunk_idx in range(num_chunks_per_hdf5_dataset):
            method_name = ("_perform_mra_of_input_ml_data_instance_chunk"
                           "_and_save_to_output_file")
            method_alias = getattr(self, method_name)
            kwargs = {"reinterpolation_is_required": \
                      reinterpolation_is_required,
                      "chunk_idx": \
                      chunk_idx,
                      "input_file_obj": \
                      input_file_obj,
                      "output_file_obj": \
                      output_file_obj}
            method_alias(**kwargs)

        input_file_obj.close()
        output_file_obj.close()

        return None



    def _perform_mra_of_input_ml_data_instance_chunk_and_save_to_output_file(
            self,
            reinterpolation_is_required,
            chunk_idx,
            input_file_obj,
            output_file_obj):
        method_alias = self._load_and_unnormalize_data_chunks
        kwargs = {"reinterpolation_is_required": reinterpolation_is_required,
                  "chunk_idx": chunk_idx,
                  "input_file_obj": input_file_obj}
        unnormalized_input_data_chunks = method_alias(**kwargs)

        key = self._ml_data_dict_key_subset_1[0]

        kwargs = \
            {"interpolated_boundary_pt_sets": \
             unnormalized_input_data_chunks[key],
             "num_pts_in_reinterpolation": \
             self._N_dot}
        reinterpolated_principal_disk_boundary_pt_sets_from_chunk = \
            (_reinterpolate_boundary_pt_sets(**kwargs)
             if reinterpolation_is_required
             else unnormalized_input_data_chunks[key])

        kwargs = \
            {"principal_disk_boundary_pt_sets": \
             reinterpolated_principal_disk_boundary_pt_sets_from_chunk}
        mra_results_of_principal_disk_boundary_pt_sets_from_chunk = \
            (_perform_mra_of_principal_disk_boundary_pt_sets(**kwargs)
             if reinterpolation_is_required
             else unnormalized_input_data_chunks.copy())

        kwargs = {"chunk_idx": \
                  chunk_idx,
                  "unnormalized_output_data_chunks": \
                  mra_results_of_principal_disk_boundary_pt_sets_from_chunk,
                  "output_file_obj": \
                  output_file_obj}
        self._save_unnormalized_output_data_chunks(**kwargs)

        return None



    def _load_and_unnormalize_data_chunks(self,
                                          reinterpolation_is_required,
                                          chunk_idx,
                                          input_file_obj):
        ml_data_dict_key_subset_5 = (self._ml_data_dict_key_subset_1
                                     if reinterpolation_is_required
                                     else self._ml_data_dict_key_subset_3)

        unnormalized_input_data_chunks = dict()
        for key in ml_data_dict_key_subset_5:
            hdf5_dataset_path = key
            method_alias = self._load_and_unnormalize_data_chunk
            kwargs = {"chunk_idx": chunk_idx,
                      "input_hdf5_dataset": input_file_obj[hdf5_dataset_path]}
            unnormalized_input_data_chunks[key] = method_alias(**kwargs)

        return unnormalized_input_data_chunks



    def _load_and_unnormalize_data_chunk(self, chunk_idx, input_hdf5_dataset):
        normalized_input_data_chunk = self._load_data_chunk(chunk_idx,
                                                            input_hdf5_dataset)

        kwargs = {"input_hdf5_dataset": input_hdf5_dataset,
                  "input_data_chunk": normalized_input_data_chunk}
        self._check_values_of_input_data_chunk(**kwargs)

        input_normalization_weight = \
            input_hdf5_dataset.attrs["normalization_weight"]
        input_normalization_bias = \
            input_hdf5_dataset.attrs["normalization_bias"]

        unnormalized_input_data_chunk = \
            ((normalized_input_data_chunk-input_normalization_bias)
             / input_normalization_weight)

        return unnormalized_input_data_chunk



    def _save_unnormalized_output_data_chunks(self,
                                              chunk_idx,
                                              unnormalized_output_data_chunks,
                                              output_file_obj):
        for key in unnormalized_output_data_chunks:
            hdf5_dataset_path = key
            kwargs = {"starting_idx_offset": \
                      self._ml_data_instance_idx_offset,
                      "chunk_idx": \
                      chunk_idx,
                      "max_num_ml_data_instances_per_chunk": \
                      self._max_num_ml_data_instances_per_chunk,
                      "data_chunk": \
                      unnormalized_output_data_chunks[key],
                      "output_hdf5_dataset": \
                      output_file_obj[hdf5_dataset_path]}
            _save_data_chunk(**kwargs)

        return None



    def _normalize_remaining_unnormalized_data_in_output_file(
            self, output_ml_dataset_filename):
        kwargs = {"path_to_ml_dataset": output_ml_dataset_filename}
        self._ml_data_normalizer._normalize_mra_in_ml_dataset_file(**kwargs)

        return None



def _is_not_power_of_2(integer):
    kwargs = locals()
    module_alias = emicroml.modelling._common
    func_name = "_is_not_power_of_2"
    func_alias = getattr(module_alias, func_name)
    result = func_alias(**kwargs)

    return result



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLDataShapeAnalyzer
class _MLDataShapeAnalyzer(_cls_alias):
    def __init__(self, for_constructing_ml_dataset, for_splitting_ml_dataset):
        self._popping_key = "principal_disk_boundary_pt_sets"
        self._for_constructing_ml_dataset = for_constructing_ml_dataset
        self._for_splitting_ml_dataset = for_splitting_ml_dataset

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



    def _hdf5_dataset_path_to_shape_map_for_ml_dataset_combo(
            self, input_ml_dataset_filenames):
        map_alias = self._ml_data_dict_key_to_shape_template_map
        popping_key = self._popping_key
        popped_map_alias_elem = map_alias.pop(popping_key)

        self._j_dashv = self._calc_j_dashv(input_ml_dataset_filenames)

        kwargs = \
            {"input_ml_dataset_filenames": input_ml_dataset_filenames}
        method_alias = \
            super()._hdf5_dataset_path_to_shape_map_for_ml_dataset_combo
        hdf5_dataset_path_to_shape_map = \
            method_alias(**kwargs)

        hdf5_dataset_path_to_shape_map[popping_key] = 2**self._j_dashv

        map_alias[popping_key] = popped_map_alias_elem

        return hdf5_dataset_path_to_shape_map



    def _calc_j_dashv(self, input_ml_dataset_filenames):
        wavelet_names = _generate_accepted_wavelet_names()
        wavelets = tuple(pywt.Wavelet(wavelet_name)
                     for wavelet_name in wavelet_names)

        j_dashv = _calc_j_dashv_candidate_lower_limit()

        for input_ml_dataset_filename in input_ml_dataset_filenames:
            kwargs = {"path_to_ml_dataset": input_ml_dataset_filename,
                      "hdf5_dataset_path": self._popping_key,
                      "read_only": True}
            hdf5_dataset = self._get_hdf5_dataset(**kwargs)

            j = np.log2(hdf5_dataset.shape[1])
            j = (np.round(j)
                 if (np.isclose(j, round(j)))
                 else np.ceil(j)).item()
            j = round(j)

            j_dashv = max(j_dashv, j)

            hdf5_dataset.file.close()

        return j_dashv



    def _hdf5_dataset_path_to_shape_map_of_ml_dataset_file(self,
                                                           path_to_ml_dataset):
        kwargs = \
            {"path_to_ml_dataset": path_to_ml_dataset}
        hdf5_dataset_path_to_shape_map = \
            super()._hdf5_dataset_path_to_shape_map_of_ml_dataset_file(**kwargs)

        for_constructing_ml_dataset = self._for_constructing_ml_dataset
        for_splitting_ml_dataset = self._for_splitting_ml_dataset

        if for_constructing_ml_dataset or for_splitting_ml_dataset:
            hdf5_dataset_path = "principal_disk_boundary_pt_sets"
            key = hdf5_dataset_path
            hdf5_dataset_shape = hdf5_dataset_path_to_shape_map[key]

            j_dashv_candidate_lower_limit = \
                _calc_j_dashv_candidate_lower_limit()

            j = np.log2(hdf5_dataset.shape[1])
            j = (np.round(j)
                 if (np.isclose(j, round(j)))
                 else np.ceil(j)).item()
            j = round(j)

            if (_is_not_power_of_2(hdf5_dataset_shape[1])
                or j < j_dashv_candidate_lower_limit):
                unformatted_err_msg = _ml_data_shape_analyzer_err_msg_1
                format_args = (hdf5_dataset_path,
                               path_to_ml_dataset,
                               j_dashv_candidate_lower_limit)
                err_msg = unformatted_err_msg.format(*format_args)
                raise ValueError(err_msg)

        return hdf5_dataset_path_to_shape_map



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
        if ("image" in key) or ("overlap_map" in key) or ("support" in key):
            axes_labels_of_hdf5_dataset = ("cropped_cbed pattern idx",
                                           "row",
                                           "col")
        elif "box" in key:
            axes_labels_of_hdf5_dataset = ("cropped cbed pattern idx",
                                           "box side idx")
        elif key == "principal_disk_boundary_pt_sets":
            axes_labels_of_hdf5_dataset = ("cropped cbed pattern idx",
                                           "pt idx")
        else:
            axes_labels_of_hdf5_dataset = ("cropped cbed pattern idx",
                                           "coeff idx")
        
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

    param_name_subset = ("num_cropped_cbed_patterns",
                         "cropped_cbed_pattern_generator")

    global_symbol_table = globals()
    for param_name in param_name_subset:
        func_name = "_check_and_convert_" + param_name
        func_alias = global_symbol_table[func_name]
        params[param_name] = func_alias(params)

    return params



def _check_and_convert_num_cropped_cbed_patterns(params):
    obj_name = "num_cropped_cbed_patterns"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}    
    num_cropped_cbed_patterns = czekitout.convert.to_positive_int(**kwargs)

    return num_cropped_cbed_patterns



def _check_and_convert_cropped_cbed_pattern_generator(params):
    obj_name = "cropped_cbed_pattern_generator"
    obj = params[obj_name]

    if obj is None:
        cls_alias = _DefaultCroppedCBEDPatternGenerator
        kwargs = {"num_pixels_across_each_cbed_pattern": \
                  _default_num_pixels_across_each_cbed_pattern,
                  "max_num_disks_in_any_cbed_pattern": \
                  _default_max_num_disks_in_any_cbed_pattern,
                  "rng_seed": \
                  _default_rng_seed,
                  "sampling_grid_dims_in_pixels": \
                  _default_sampling_grid_dims_in_pixels,
                  "least_squares_alg_params": \
                  _default_least_squares_alg_params,
                  "device_name": \
                  _default_device_name,
                  "num_pixels_across_each_expected_cropping_window": \
                  _default_num_pixels_across_each_expected_cropping_window,
                  "skip_validation_and_conversion": \
                  False}
        cropped_cbed_pattern_generator = cls_alias(**kwargs)
    else:
        cropped_cbed_pattern_generator = obj
    
    return cropped_cbed_pattern_generator



_module_alias = emicroml.modelling.cbed._common
_default_cropped_cbed_pattern_generator = None
_default_num_cropped_cbed_patterns = _module_alias._default_num_cbed_patterns
_default_output_filename = _module_alias._default_output_filename



def _generate_and_save_ml_dataset(cropped_cbed_pattern_generator,
                                  max_num_ml_data_instances_per_file_update,
                                  num_cropped_cbed_patterns,
                                  output_filename,
                                  start_time):
    kwargs = \
        {"cropped_cbed_pattern_generator": cropped_cbed_pattern_generator}
    unnormalized_ml_data_instance_generator = \
        _UnnormalizedMLDataInstanceGenerator(**kwargs)

    kwargs = {"max_num_ml_data_instances_per_file_update": \
              max_num_ml_data_instances_per_file_update}
    ml_data_normalizer = _MLDataNormalizer(**kwargs)

    num_ml_data_instances = num_cropped_cbed_patterns

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

    kwargs = {"for_constructing_ml_dataset": False,
              "for_splitting_ml_dataset": False}
    ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

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

    kwargs = {"for_constructing_ml_dataset": False,
              "for_splitting_ml_dataset": True}
    ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

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
              "split_ratio": \
              split_ratio}
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



def _get_num_pixels_across_each_cropped_cbed_pattern(path_to_ml_dataset,
                                                     ml_data_shape_analyzer):
    obj_alias = \
        ml_data_shape_analyzer
    method_alias = \
        obj_alias._hdf5_dataset_path_to_shape_map_of_ml_dataset_file
    hdf5_dataset_path_to_shape_map = \
        method_alias(path_to_ml_dataset)

    hdf5_dataset_shape = \
        hdf5_dataset_path_to_shape_map["cropped_cbed_pattern_images"]
    num_pixels_across_each_cropped_cbed_pattern = \
        hdf5_dataset_shape[-1]

    return num_pixels_across_each_cropped_cbed_pattern



def _check_and_convert_normalize_normalizable_elems_in_ml_data_dict_params(
        params):
    original_params = params
    params = params.copy()

    kwargs = {"for_constructing_ml_dataset": False,
              "for_splitting_ml_dataset": False}
    ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

    params["ml_data_shape_analyzer"] = \
        ml_data_shape_analyzer
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
    all_valid_ml_data_dict_keys = _generate_all_valid_ml_data_dict_keys()

    for key_1 in all_valid_ml_data_dict_keys:
        if key_1 not in extrema_cache:
            continue
        for key_2 in ("min", "max"):
                extrema_cache[key_1][key_2] = int(key_2 == "max")

    return None



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

    kwargs = {"for_constructing_ml_dataset": False,
              "for_splitting_ml_dataset": False}
    ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

    params["ml_data_shape_analyzer"] = \
        ml_data_shape_analyzer
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



def _check_and_convert_ml_data_dict(params):
    params = params.copy()

    kwargs = {"for_constructing_ml_dataset": False,
              "for_splitting_ml_dataset": False}
    ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

    params["name_of_obj_alias_of_ml_data_dict"] = "ml_data_dict"
    params["ml_data_normalizer"] = _generate_default_ml_data_normalizer()
    params["target_numerical_data_container_cls"] = None
    params["target_device"] = None
    params["variable_axis_size_dict"] = None
    params["ml_data_shape_analyzer"] = ml_data_shape_analyzer
    params["ml_data_type_validator"] = _MLDataTypeValidator()
    params["normalizable_elems_are_normalized"] = False
    params["ml_data_value_validator"] = _MLDataValueValidator()

    module_alias = emicroml.modelling._common
    func_alias = module_alias._check_and_convert_ml_data_dict
    ml_data_dict = func_alias(params)

    return ml_data_dict



def _check_and_convert_ml_data_dict_to_signals_params(params):
    original_params = params
    params = params.copy()

    params["bounding_box_marker_style_kwargs"] = \
        _check_and_convert_bounding_box_marker_style_kwargs(params)
    params["boundary_pt_marker_style_kwargs"] = \
        _check_and_convert_boundary_pt_marker_style_kwargs(params)

    kwargs = {"obj": params["ml_data_dict"], "obj_name": "ml_data_dict"}
    ml_data_dict = czekitout.convert.to_dict(**kwargs).copy()
    params["ml_data_dict"] = ml_data_dict

    device_name = "cpu"

    params = \
        {"cropped_cbed_pattern_images": \
         ml_data_dict.get("cropped_cbed_pattern_images", None),
         "name_of_obj_alias_of_cropped_cbed_pattern_images": \
         "ml_data_dict['cropped_cbed_pattern_images']",
         "target_device": \
         _get_device(device_name),
         **params}
    cropped_cbed_pattern_images = \
        _check_and_convert_cropped_cbed_pattern_images(params)
    params["ml_data_dict"]["cropped_cbed_pattern_images"] = \
        cropped_cbed_pattern_images

    params["data_chunk_dims_are_to_be_expanded_temporarily"] = \
        False
    params["expected_ml_data_dict_keys"] = \
        ("cropped_cbed_pattern_images",)
    params["ml_data_dict"] = \
        _check_and_convert_ml_data_dict(params)

    for key in tuple(params.keys()):
        if key not in original_params:
            del params[key]
    
    return params



def _check_and_convert_bounding_box_marker_style_kwargs(params):
    obj_name = "bounding_box_marker_style_kwargs"
    obj = params[obj_name]

    accepted_types = (dict, type(None))

    current_func_name = "_check_and_convert_bounding_box_marker_style_kwargs"

    if isinstance(obj, accepted_types[-1]):
        bounding_box_marker_style_kwargs = obj
    else:
        try:
            kwargs = {"obj": obj, "obj_name": obj_name}
            obj = czekitout.convert.to_dict(**kwargs)
        except:
            kwargs = {"obj": obj,
                      "obj_name": obj_name,
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)

        invalid_keys = ("offset_transform",
                        "transform",
                        "shift",
                        "plot_on_signal",
                        "name",
                        "ScalarMappable_array",
                        "offsets",
                        "heights",
                        "widths",
                        "angles")

        if any(key in obj for key in invalid_keys):
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise KeyError(err_msg)        

        try:
            kwargs = {"bounding_box": np.array((0.25, 0.75, 0.25, 0.75)),
                      "bounding_box_marker_style_kwargs": obj}
            bounding_box_marker = _generate_bounding_box_marker(**kwargs)

            bounding_box_marker_style_kwargs = obj
        except KeyError:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise KeyError(err_msg)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise ValueError(err_msg)

    return bounding_box_marker_style_kwargs



def _generate_bounding_box_marker(bounding_box,
                                  bounding_box_marker_style_kwargs):
    L_box, R_box, B_box, T_box = bounding_box

    q_x_c_box = (R_box+L_box)/2.0
    q_y_c_box = (B_box+T_box)/2.0

    kwargs = {"offsets": (q_x_c_box, q_y_c_box),
              "widths": (R_box-L_box),
              "heights": (B_box-T_box),
              **bounding_box_marker_style_kwargs}
    bounding_box_marker = hs.plot.markers.Rectangles(**kwargs)

    return bounding_box_marker



def _check_and_convert_boundary_pt_marker_style_kwargs(params):
    obj_name = "boundary_pt_marker_style_kwargs"
    obj = params[obj_name]

    accepted_types = (dict, type(None))

    current_func_name = "_check_and_convert_boundary_pt_marker_style_kwargs"

    if isinstance(obj, accepted_types[-1]):
        boundary_pt_marker_style_kwargs = obj
    else:
        try:
            kwargs = {"obj": obj, "obj_name": obj_name}
            obj = czekitout.convert.to_dict(**kwargs).copy()
        except:
            kwargs = {"obj": obj,
                      "obj_name": obj_name,
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)

        invalid_keys = ("offset_transform",
                        "transform",
                        "shift",
                        "plot_on_signal",
                        "name",
                        "ScalarMappable_array",
                        "offsets",
                        "sizes")

        if any(key in obj for key in invalid_keys):
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise KeyError(err_msg)
        
        try:
            kwargs = {"boundary_pt_set": np.array(((0.5, 0.5),)),
                      "boundary_pt_marker_style_kwargs": obj}
            boundary_pt_marker_set = _generate_boundary_pt_marker_set(**kwargs)

            boundary_pt_marker_style_kwargs = obj
        except KeyError:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise KeyError(err_msg)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise ValueError(err_msg)

    return boundary_pt_marker_style_kwargs



def _generate_boundary_pt_marker_set(boundary_pt_set,
                                     boundary_pt_marker_style_kwargs):
    single_dim_slice = boundary_pt_marker_style_kwargs.pop("single_dim_slice",
                                                           slice(None))

    kwargs = {"offsets": boundary_pt_set[single_dim_slice, :],
              **boundary_pt_marker_style_kwargs}
    boundary_pt_marker_set = hs.plot.markers.Points(**kwargs)

    boundary_pt_marker_style_kwargs["single_dim_slice"] = single_dim_slice

    return boundary_pt_marker_set



def _check_and_convert_cropped_cbed_pattern_images(params):
    obj_name = "cropped_cbed_pattern_images"
    obj = params[obj_name]

    name_of_obj_alias_of_cropped_cbed_pattern_images = \
        params.get("name_of_obj_alias_of_cropped_cbed_pattern_images", obj_name)
    target_device = \
        params.get("target_device", None)

    kwargs = {"numerical_data_container": \
              obj,
              "name_of_obj_alias_of_numerical_data_container": \
              name_of_obj_alias_of_cropped_cbed_pattern_images,
              "target_numerical_data_container_cls": \
              torch.Tensor,
              "target_device": \
              None}
    obj = _convert_numerical_data_container(**kwargs)

    current_func_name = "_check_and_convert_cropped_cbed_pattern_images"

    if len(obj.shape) != 3:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        args = (name_of_obj_alias_of_cropped_cbed_pattern_images,)
        err_msg = unformatted_err_msg.format(*args)
        raise ValueError(err_msg)

    kwargs = {"image_stack": obj}
    cropped_cbed_pattern_images = _min_max_normalize_image_stack(**kwargs)

    return cropped_cbed_pattern_images



def _min_max_normalize_image_stack(image_stack):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_name = "_min_max_normalize_image_stack"
    func_alias = getattr(module_alias, func_name)
    normalized_image_stack = func_alias(**kwargs)

    return normalized_image_stack



def _ml_data_dict_to_signals(ml_data_dict,
                             bounding_box_marker_style_kwargs,
                             boundary_pt_marker_style_kwargs):
    kwargs = \
        {"ml_data_dict": ml_data_dict, "device_name": "cpu"}
    cropped_cbed_pattern_images = \
        _get_cropped_cbed_pattern_images_from_ml_data_dict(**kwargs)
    cropped_cbed_pattern_images = \
        cropped_cbed_pattern_images.numpy(force=True)

    kwargs = \
        {"ml_data_dict": ml_data_dict}
    cropped_disk_overlap_maps = \
        _get_cropped_disk_overlap_maps_from_ml_data_dict(**kwargs)
    cropped_principal_disk_supports = \
        _get_cropped_principal_disk_supports_from_ml_data_dict(**kwargs)
    principal_disk_bounding_boxes = \
        _get_principal_disk_bounding_boxes_from_ml_data_dict(**kwargs)
    principal_disk_boundary_pt_sets = \
        _generate_principal_disk_boundary_pt_sets_from_ml_data_dict(**kwargs)

    signals = tuple()
    global_symbol_table = globals()
    for cropped_cbed_pattern_idx, _ in enumerate(cropped_cbed_pattern_images):
        func_name = "_construct_signal_using_objs_extracted_from_ml_data_dict"
        func_alias = global_symbol_table[func_name]
        kwargs = {"cropped_cbed_pattern_image": \
                  cropped_cbed_pattern_images[cropped_cbed_pattern_idx],
                  "cropped_disk_overlap_map": \
                  cropped_disk_overlap_maps[cropped_cbed_pattern_idx],
                  "cropped_principal_disk_support": \
                  cropped_principal_disk_supports[cropped_cbed_pattern_idx],
                  "principal_disk_bounding_box": \
                  principal_disk_bounding_boxes[cropped_cbed_pattern_idx],
                  "bounding_box_marker_style_kwargs": \
                  bounding_box_marker_style_kwargs,
                  "principal_disk_boundary_pt_set": \
                  principal_disk_boundary_pt_sets[cropped_cbed_pattern_idx],
                  "boundary_pt_marker_style_kwargs": \
                  boundary_pt_marker_style_kwargs}
        signal = func_alias(**kwargs)
        signals += (signal,)

    return signals



def _get_cropped_cbed_pattern_images_from_ml_data_dict(ml_data_dict,
                                                       device_name):
    key = "cropped_cbed_pattern_images"

    module_alias = \
        emicroml.modelling._common
    convert_numerical_data_container = \
        module_alias._convert_numerical_data_container

    kwargs = {"numerical_data_container": \
              ml_data_dict[key],
              "name_of_obj_alias_of_numerical_data_container": \
              "ml_data_dict['{}']".format(key),
              "target_numerical_data_container_cls": \
              torch.Tensor,
              "target_device": \
              _get_device(device_name)}
    cropped_cbed_pattern_images = convert_numerical_data_container(**kwargs)

    return cropped_cbed_pattern_images



def _get_cropped_disk_overlap_maps_from_ml_data_dict(ml_data_dict):
    key = "cropped_disk_overlap_maps"

    module_alias = \
        emicroml.modelling._common
    convert_numerical_data_container = \
        module_alias._convert_numerical_data_container

    try:
        kwargs = \
            {"numerical_data_container": \
             ml_data_dict[key],
             "name_of_obj_alias_of_numerical_data_container": \
             "ml_data_dict['{}']".format(key),
             "target_numerical_data_container_cls": \
             np.ndarray,
             "target_device": \
             None}
        cropped_disk_overlap_maps = \
            convert_numerical_data_container(**kwargs)
    except:
        cropped_cbed_pattern_images = \
            ml_data_dict["cropped_cbed_pattern_images"]
        num_cropped_cbed_patterns = \
            cropped_cbed_pattern_images.shape[0]
        cropped_disk_overlap_maps = \
            (None,) * num_cropped_cbed_patterns

    return cropped_disk_overlap_maps



def _get_cropped_principal_disk_supports_from_ml_data_dict(ml_data_dict):
    key = "cropped_principal_disk_supports"

    module_alias = \
        emicroml.modelling._common
    convert_numerical_data_container = \
        module_alias._convert_numerical_data_container

    try:
        kwargs = \
            {"numerical_data_container": \
             ml_data_dict[key],
             "name_of_obj_alias_of_numerical_data_container": \
             "ml_data_dict['{}']".format(key),
             "target_numerical_data_container_cls": \
             np.ndarray,
             "target_device": \
             None}
        cropped_principal_disk_supports = \
            convert_numerical_data_container(**kwargs)
    except:
        cropped_cbed_pattern_images = \
            ml_data_dict["cropped_cbed_pattern_images"]
        num_cropped_cbed_patterns = \
            cropped_cbed_pattern_images.shape[0]
        cropped_principal_disk_supports = \
            (None,) * num_cropped_cbed_patterns

    return cropped_principal_disk_supports



def _get_principal_disk_bounding_boxes_from_ml_data_dict(ml_data_dict):
    key = "principal_disk_bounding_boxes"

    module_alias = \
        emicroml.modelling._common
    convert_numerical_data_container = \
        module_alias._convert_numerical_data_container

    try:
        kwargs = \
            {"numerical_data_container": \
             ml_data_dict[key],
             "name_of_obj_alias_of_numerical_data_container": \
             "ml_data_dict['{}']".format(key),
             "target_numerical_data_container_cls": \
             np.ndarray,
             "target_device": \
             None}
        principal_disk_bounding_boxes = \
            convert_numerical_data_container(**kwargs)
    except:
        cropped_cbed_pattern_images = \
            ml_data_dict["cropped_cbed_pattern_images"]
        num_cropped_cbed_patterns = \
            cropped_cbed_pattern_images.shape[0]
        principal_disk_bounding_boxes = \
            (None,) * num_cropped_cbed_patterns

    return principal_disk_bounding_boxes



def _get_principal_disk_boundary_pt_sets_from_ml_data_dict(ml_data_dict):
    key = "principal_disk_boundary_pt_sets"

    module_alias = \
        emicroml.modelling._common
    convert_numerical_data_container = \
        module_alias._convert_numerical_data_container

    try:
        kwargs = \
            {"numerical_data_container": \
             ml_data_dict[key],
             "name_of_obj_alias_of_numerical_data_container": \
             "ml_data_dict['{}']".format(key),
             "target_numerical_data_container_cls": \
             np.ndarray,
             "target_device": \
             None}
        principal_disk_boundary_pt_sets = \
            convert_numerical_data_container(**kwargs)
    except:
        cropped_cbed_pattern_images = \
            ml_data_dict["cropped_cbed_pattern_images"]
        num_cropped_cbed_patterns = \
            cropped_cbed_pattern_images.shape[0]
        principal_disk_boundary_pt_sets = \
            (None,) * num_cropped_cbed_patterns

    return principal_disk_boundary_pt_sets



def _construct_signal_using_objs_extracted_from_ml_data_dict(
        cropped_cbed_pattern_image,
        cropped_disk_overlap_map,
        cropped_principal_disk_support,
        principal_disk_bounding_box,
        bounding_box_marker_style_kwargs,
        principal_disk_boundary_pt_set,
        boundary_pt_marker_style_kwargs):
    kwargs = {"cropped_cbed_pattern_image": cropped_cbed_pattern_image,
              "cropped_disk_overlap_map": cropped_disk_overlap_map}
    inferred_illumination_support = _infer_illumination_support(**kwargs)

    signal_data_shape = (4,) + cropped_cbed_pattern_image.shape

    signal_data = np.zeros(signal_data_shape,
                           dtype=cropped_cbed_pattern_image.dtype)
    signal_data[0] = cropped_cbed_pattern_image
    signal_data[1] = inferred_illumination_support
    if cropped_disk_overlap_map is not None:
        signal_data[2] = cropped_disk_overlap_map
    if cropped_principal_disk_support is not None:
        signal_data[3] = cropped_principal_disk_support

    signal = hyperspy.signals.Signal2D(data=signal_data)

    _update_signal_axes(signal)

    if ((principal_disk_bounding_box is not None):
        and (bounding_box_marker_style_kwargs is not None)):
        kwargs = {"bounding_box": \
                  principal_disk_bounding_box,
                  "bounding_box_marker_style_kwargs": \
                  bounding_box_marker_style_kwargs}
        bounding_box_marker = _generate_bounding_box_marker(**kwargs)
        
        signal.add_marker(bounding_box_marker, permanent=True)

    if ((principal_disk_boundary_pt_set is not None)
        and (boundary_pt_marker_style_kwargs)):
        kwargs = {"boundary_pt_set": \
                  principal_disk_boundary_pt_set,
                  "boundary_pt_marker_style_kwargs": \
                  boundary_pt_marker_style_kwargs}
        boundary_pt_marker_set = _generate_boundary_pt_marker_set(**kwargs)

        signal.add_marker(boundary_pt_marker_set, permanent=True)

    return signal



def _infer_illumination_support(cropped_cbed_pattern_image,
                                cropped_disk_overlap_map):
    inferred_illumination_support = \
        (cropped_cbed_pattern_image != 0).numpy(force=True)
    if cropped_disk_overlap_map is not None:
        inferred_illumination_support += \
            (cropped_disk_overlap_map > 0)

    return inferred_illumination_support



def _update_signal_axes(signal):
    num_axes = (len(signal.axes_manager.navigation_shape)
                + len(signal.axes_manager.signal_shape))

    partial_axis_label = \
        r"fractional {} coordinate of cropped fake CBED pattern axis"

    sizes = (signal.axes_manager.navigation_shape
             + signal.axes_manager.signal_shape)
    scales = ((1,)*len(signal.axes_manager.navigation_shape)
              + (1/sizes[-2], -1/sizes[-1]))
    offsets = ((0,)*len(signal.axes_manager.navigation_shape)
               + (0.5*scales[-2], 1+(1-0.5)*scales[-1]))
    axes_labels = (r"cropped fake CBED pattern attribute",
                   partial_axis_label.format("horizontal"),
                   partial_axis_label.format("vertical"))
    units = ("dimensionless",)*num_axes

    for axis_idx in range(num_axes):
        kwargs = {"size": sizes[axis_idx],
                  "scale": scales[axis_idx],
                  "offset": offsets[axis_idx],
                  "units": units[axis_idx],
                  "name": axes_labels[axis_idx]}
        axis = hyperspy.axes.UniformDataAxis(**kwargs)
        signal.axes_manager[axis_idx].update_from(axis)
        signal.axes_manager[axis_idx].name = axis.name

    return None



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
_default_bounding_box_marker_style_kwargs = \
    None
_default_boundary_pt_marker_style_kwargs = \
    None



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

        kwargs = {"for_constructing_ml_dataset": False,
                  "for_splitting_ml_dataset": False}
        ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

        func_alias = _get_num_pixels_across_each_cropped_cbed_pattern
        kwargs = {"path_to_ml_dataset": self_core_attrs["path_to_ml_dataset"],
                  "ml_data_shape_analyzer": ml_data_shape_analyzer}
        self._num_pixels_across_each_cropped_cbed_pattern = func_alias(**kwargs)

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

        kwargs = {"for_constructing_ml_dataset": True,
                  "for_splitting_ml_dataset": False}
        ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._TorchMLDataset
        kwargs = {"path_to_ml_dataset": \
                  self_core_attrs["path_to_ml_dataset"],
                  "ml_data_normalization_weights_and_biases_loader": \
                  ml_data_normalization_weights_and_biases_loader,
                  "ml_data_type_validator": \
                  _MLDataTypeValidator(),
                  "ml_data_shape_analyzer": \
                  ml_data_shape_analyzer,
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
            bounding_box_marker_style_kwargs=\
            _default_bounding_box_marker_style_kwargs,
            boundary_pt_marker_style_kwargs=\
            _default_boundary_pt_marker_style_kwargs):
        r"""Return a subset of the machine learning data instances as a sequence
        of Hyperspy signals.

        See the documentation for the classes
        :class:`fakecbed.discretized.CroppedCBEDPattern`, and
        :class:`hyperspy._signals.signal2d.Signal2D` for discussions on cropped 
        "fake" CBED patterns, and Hyperspy signals respectively.

        For each machine learning (ML) data instance in the subset of ML
        instances, the ML data instance is expected to represent a subset of the
        properties of a cropped CBED pattern, that could be represented
        hypothetically as an instance ``cropped_fake_cbed_pattern`` of the class
        :class:`fakecbed.discretized.CroppedCBEDPattern`. The signal data of the
        Hyperspy signal representation ``ml_data_instance_as_signal`` of the ML
        data instance is equal to ``cropped_fake_cbed_pattern.signal.data[:4]``,
        where ``cropped_fake_cbed_pattern.signal.data[1]``, which stores the
        illumination support of the cropped fake CBED pattern, is inferred from
        the features of the ML data instance. The navigation space axes of
        ``ml_data_instance_as_signal`` are identical to those of
        ``cropped_fake_cbed_pattern.signal.inav[:4]``, and the signal space axes
        of ``ml_data_instance_as_signal`` are chosen such that coordinates in
        this space are consistent with the fractional coordinates of the cropped
        CBED pattern. If the parameter ``bounding_box_marker_style_kwargs`` is
        not set to ``None``, but a valid dictionary (see parameter descriptions
        below for details), the bounding box of the principal disk of the
        cropped fake CBED pattern is permanently added to
        ``ml_data_instance_as_signal`` as a Hyperspy marker, with style
        properties according to ``bounding_box_marker_style_kwargs``. Similarly,
        if the parameter ``boundary_pt_marker_style_kwargs`` is not set to
        ``None``, but a valid dictionary, points on the boundary of the
        principal disk of the cropped fake CBED pattern are permanently added to
        ``ml_data_instance_as_signal`` as Hyperspy markers, with style
        properties according to ``boundary_pt_marker_style_kwargs``. Note that
        apart from the default signal metadata, no other metadata is added to
        the signal. The title of each Hyperspy signal is ``"Cropped CBED
        Intensity Pattern"``.

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
        bounding_box_marker_style_kwargs : `None` | `dict`, optional
            If ``bounding_box_marker_style_kwargs`` is set to ``None``, then the
            bounding boxes of the principal disks of the cropped fake CBED
            patterns corresponding to the ML data instances in the subset are
            not accessed, nor are they added to their corresponding Hyperspy
            signal representations of said ML data instances. 
            
            Otherwise, if ``bounding_box_marker_style_kwargs`` is set to a
            dictionary, then the bounding boxes are added permanently to their
            corresponding Hyperspy signals as instances of the
            :class:`hyperspy.api.plot.markers.Rectangles` class (i.e. as
            rectangular Hyperspy markers), with stylistic properties determined
            by ``bounding_box_marker_style_kwargs``. All valid dictionary items
            are optional. A valid dictionary item is any keyword argument for
            the constructor of the class
            :class:`hyperspy.api.plot.markers.Rectangles` other than::

            * ``"offset_transform"``
            * ``"transform"``
            * ``"shift"``
            * ``"plot_on_signal"``
            * ``"name"``
            * ``"ScalarMappable_array"``
            * ``"offsets"``
            * ``"heights"``
            * ``"widths"``
            * ``"angles"``

            The default value of each valid dictionary item is that of the
            corresponding keyword argument for the constructor. These valid
            dictionary items are used directly to construct rectangular Hyperspy
            markers.
        boundary_pt_marker_style_kwargs : `None` | `dict`, optional            
            If ``boundary_pt_marker_style_kwargs`` is set to ``None``, then no
            points on the boundaries of the principal disks of the cropped fake
            CBED patterns corresponding to the ML data instances in the subset
            are accessed, nor are any points added to their corresponding
            Hyperspy signal representations of said ML data
            instances. 

            Otherwise, if ``boundary_pt_marker_style_kwargs`` is set to a
            dictionary, then subsets of the points on the boundaries are added
            permanently to their corresponding Hyperspy signals as instances of
            the :class:`hyperspy.api.plot.markers.Points` class (i.e. as
            collections of point/circular Hyperspy markers), with stylistic
            properties determined by ``boundary_pt_marker_style_kwargs``. All
            valid dictionary items are optional. One of the valid dictionary
            items is a `slice` object stored in
            ``boundary_pt_marker_style_kwargs["single_dim_slice"]``, which
            controls what subset of available points are added as a collection
            of markers, for each ML data instance. For each ML data instance,
            the boundary point indices are indexed from ``0`` to
            ``total_num_pts_per_ml_data_instance-1``, where
            ``total_num_pts_per_ml_data_instance`` is the total number of
            sampled boundary points per ML data instance.
            ``tuple(range(total_num_pts_per_ml_data_instance))[boundary_pt_marker_style_kwargs["single_dim_slice"]]``
            yields the indices of the points to add as markers. It is
            recommended that users limit the number of points that they add to
            the signals as this can slow down signal plotting if the number is
            too large. The default value of
            ``boundary_pt_marker_style_kwargs["single_dim_slice"]`` is
            ``slice(None)``.

            The remaining valid dictionary items are any keyword arguments for
            the constructor of the class
            :class:`hyperspy.api.plot.markers.Points` other than::

            * ``"offset_transform"``
            * ``"transform"``
            * ``"shift"``
            * ``"plot_on_signal"``
            * ``"name"``
            * ``"ScalarMappable_array"``
            * ``"offsets"``
            * ``"sizes"``

            The default value of each valid dictionary item listed immediately
            above is that of the corresponding keyword argument for the
            constructor. These valid dictionary items are used directly to
            construct point/circular Hyperspy markers.

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
        bounding_box_marker_style_kwargs = \
            _check_and_convert_bounding_box_marker_style_kwargs(params)
        boundary_pt_marker_style_kwargs = \
            _check_and_convert_boundary_pt_marker_style_kwargs(params)

        kwargs = {"single_dim_slice": single_dim_slice,
                  "device_name": "cpu",
                  "decode": True,
                  "unnormalize_normalizable_elems": True}
        ml_data_instances = self.get_ml_data_instances(**kwargs)

        kwargs = {"ml_data_dict": \
                  ml_data_instances,
                  "bounding_box_marker_style_kwargs": \
                  bounding_box_marker_style_kwargs,
                  "boundary_pt_marker_style_kwargs": \
                  boundary_pt_marker_style_kwargs}
        ml_data_instances_as_signals = _ml_data_dict_to_signals(**kwargs)

        return ml_data_instances_as_signals



    @property
    def num_pixels_across_each_cropped_cbed_pattern(self):
        r"""`int`: The number of pixels across each imaged cropped CBED pattern
        stored in the machine learning dataset.

        Note that ``num_pixels_across_each_cropped_cbed_pattern`` should be 
        considered **read-only**.

        """
        result = self._num_pixels_across_each_cropped_cbed_pattern
        
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
        kwargs = ctor_params
        cls_alias.__init__(self, **kwargs)

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



_building_block_counts_in_stages_of_localization_net = \
    (3, 5, 2)



class _LocalizationNet(torch.nn.Module):
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
            _building_block_counts_in_stages_of_localization_net
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
                  4,
                  "mini_batch_norm_eps": \
                  self._mini_batch_norm_eps}
        distoptica_net = module_alias._DistopticaNet(**kwargs)

        return distoptica_net



    def forward(self, ml_inputs):
        enhanced_cropped_cbed_pattern_images = \
            self._get_and_enhance_cropped_cbed_pattern_images(ml_inputs)

        intermediate_tensor = enhanced_cropped_cbed_pattern_images
        output_tensor, _ = self._distoptica_net(intermediate_tensor)

        key = "principal_disk_bounding_boxes"
        ml_predictions = {"principal_disk_bounding_boxes": output_tensor}

        return ml_predictions



    def _get_and_enhance_cropped_cbed_pattern_images(self, ml_inputs):
        kwargs = \
            {"image_stack": ml_inputs["cropped_cbed_pattern_images"]}
        enhanced_cropped_cbed_pattern_images = \
            _min_max_normalize_image_stack(**kwargs)

        gamma = 0.3

        enhanced_cropped_cbed_pattern_images = \
            torch.unsqueeze(enhanced_cropped_cbed_pattern_images, dim=1)
        enhanced_cropped_cbed_pattern_images = \
            torch.pow(enhanced_cropped_cbed_pattern_images, gamma)
        enhanced_cropped_cbed_pattern_images = \
            kornia.enhance.equalize(enhanced_cropped_cbed_pattern_images)

        kwargs = {"input": enhanced_cropped_cbed_pattern_images,
                  "min": 0,
                  "max": 1}
        enhanced_cropped_cbed_pattern_images = torch.clip(**kwargs)

        return enhanced_cropped_cbed_pattern_images



def _get_wavelet_name_from_j_vdash(j_vdash):
    wavelet_name = "db"+str(2**(j_vdash-1))

    return wavelet_name



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._BasicResNetStage
class _BasicResNetStage(_cls_alias):
    def __init__(self,
                 num_input_channels,
                 max_kernel_size,
                 num_building_blocks,
                 final_activation_func,
                 mini_batch_norm_eps):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._BasicResNetStage
        cls_alias.__init__(self, ctor_params)

        return None



def _initialize_layer_weights_according_to_activation_func(activation_func,
                                                           layer):
    module_alias = emicroml.modelling._common
    func_name = "_initialize_layer_weights_according_to_activation_func"
    func_alias = getattr(module_alias, func_name)
    kwargs = {"activation_func": activation_func, "layer": layer}
    func_alias(**kwargs)

    return None



class _BottleneckBlock(torch.nn.Module):
    def __init__(self,
                 num_pixels_across_each_cbed_pattern,
                 num_input_channels,
                 num_filters_per_conv_layer,
                 j_vdash,
                 j,
                 j_dashv,
                 mini_batch_norm_eps):
        super().__init__()

        self._num_pixels_across_each_cbed_pattern = \
            num_pixels_across_each_cbed_pattern
        self._num_input_channels = \
            num_input_channels
        self._num_filters_per_conv_layer = \
            num_filters_per_conv_layer
        self._j_vdash = \
            j_vdash
        self._j = \
            j
        self._j_dashv = \
            j_dashv
        self._mini_batch_norm_eps = \
            mini_batch_norm_eps

        self._num_output_channels = 2**(j_dashv-1)

        self._conv_layers = self._generate_conv_layers()
        self._mini_batch_norms = self._generate_mini_batch_norms()
        self._fc_layer = self._generate_fc_layer()
        
        return None


    
    def _generate_conv_layers(self):
        num_conv_layers = 0 + (self._j > self._j_vdash)

        conv_layers = tuple(self._generate_conv_layer()
                            for conv_layer_idx
                            in range(num_conv_layers))
        
        conv_layers = torch.nn.ModuleList(conv_layers)

        return conv_layers



    def _generate_conv_layer(self):
        kwargs = {"in_channels": self._num_input_channels,
                  "out_channels": self._num_filters_per_conv_layer,
                  "kernel_size": 1,
                  "stride": 1,
                  "padding": 0,
                  "padding_mode": "zeros",
                  "bias": False}
        conv_layer = torch.nn.Conv2d(**kwargs)

        kwargs = {"conv_layer": conv_layer}
        self._initialize_conv_layer_weights(**kwargs)

        return conv_layer



    def _initialize_conv_layer_weights(self, conv_layer):
        kwargs = {"activation_func": torch.nn.ReLU(), "layer": conv_layer}
        _initialize_layer_weights_according_to_activation_func(**kwargs)

        return None



    def _generate_mini_batch_norms(self):
        num_mini_batch_norms = len(self._conv_layers)

        mini_batch_norms = \
            tuple(self._generate_mini_batch_norm()
                  for mini_batch_norm_idx
                  in range(num_mini_batch_norms))

        mini_batch_norms = torch.nn.ModuleList(mini_batch_norms)

        return mini_batch_norms



    def _generate_mini_batch_norm(self):
        kwargs = {"num_features": self._num_filters_per_conv_layer,
                  "eps": self._mini_batch_norm_eps}
        mini_batch_norm = torch.nn.BatchNorm2d(**kwargs)

        kwargs = {"mini_batch_norm": mini_batch_norm}
        self._initialize_mini_batch_norm_weights_and_biases(**kwargs):

        return mini_batch_norm


    
    def _initialize_mini_batch_norm_weights_and_biases(self, mini_batch_norm):
        torch.nn.init.constant_(mini_batch_norm.bias, 0)
        torch.nn.init.constant_(mini_batch_norm.weight, 1)
            
        return None



    def _generate_fc_layer(self):
        N = self._num_pixels_across_each_cbed_pattern
        C = (self._num_filters_per_conv_layer
             if (j > j_vdash)
             else self._num_input_channels)

        kwargs = {"in_features": N*N*C,
                  "out_features": self._num_output_channels,
                  "bias": True}
        fc_layer = torch.nn.Linear(**kwargs)

        self._initialize_fc_layer_weights(self, fc_layer)

        return fc_layer



    def _initialize_fc_layer_weights(self, fc_layer):
        kwargs = {"activation_func": torch.nn.ReLU(), "layer": fc_layer}
        _initialize_layer_weights_according_to_activation_func(**kwargs)

        return None



    def forward(self, X):
        Y = X
        
        for conv_layer_idx, conv_layer in enumerate(self._conv_layers):
            mini_batch_norm_idx = conv_layer_idx
            mini_batch_norm = self._mini_batch_norms[mini_batch_norm_idx]
            
            Y = conv_layer(Y)
            Y = mini_batch_norm(Y)
            Y = torch.nn.functional.relu(Y)

        fc_layer = self._fc_layer
            
        Y = torch.flatten(Y, start_dim=1)
        Y = fc_layer(Y)
        Y = torch.nn.functional.relu(Y)

        return Y



class IDWTBlock(torch.nn.Module):
    def __init__(self, j_vdash):
        super().__init__()

        wavelet_name = _get_wavelet_name_from_j_vdash(j_vdash)
        kwargs = {"wave": wavelet_name, "mode": "periodization"}
        self._idwt = pytorch_wavelets.dwt.transform1d.DWT1DInverse(**kwargs)

        return None



    def forward(self, a_j, d_j):
        a_j = torch.unsqueeze(a_j, 1)
        d_j = torch.unsqueeze(d_j, 1)
        
        a_jP1 = self._idwt((a_j, (d_j,)))
        a_jP1 = torch.squeeze(a_jP1, 1)

        return a_jP1



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._FCResidualBlock
class _FCResidualBlock(_cls_alias):
    def __init__(self,
                 num_input_channels,
                 final_activation_func,
                 mini_batch_norm_eps):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._FCResidualBlock
        cls_alias.__init__(self, ctor_params)

        return None



class _PredictionBlock(torch.nn.Module):
    def __init__(self,
                 j,
                 j_dashv,
                 mini_batch_norm_eps):
        super().__init__()

        self._j = \
            j
        self._j_dashv = \
            j_dashv
        self._mini_batch_norm_eps = \
            mini_batch_norm_eps

        self._num_input_channels = 2**(j_dashv-1)
        self._num_output_channels = 2**j

        self._fc_residual_block = self._generate_fc_residual_block()        
        self._fc_layer = self._generate_fc_layer()

        return None



    def _generate_fc_residual_block(self):
        kwargs = {"num_input_channels": self._num_input_channels,
                  "final_activation_func": torch.nn.ReLU(),
                  "mini_batch_norm_eps": self._mini_batch_norm_eps}
        fc_residual_block = self._generate_fc_residual_block()

        return fc_residual_block



    def _generate_fc_layer(self):
        kwargs = {"in_features": self._num_input_channels,
                  "out_features": self._num_output_channels,
                  "bias": True}
        fc_layer = torch.nn.Linear(**kwargs)

        kwargs = {"fc_layer": fc_layer}
        self._initialize_fc_layer_weights(**kwargs)

        return fc_layer



    def _initialize_fc_layer_weights(self, fc_layer):
        kwargs = {"activation_func": torch.nn.Identity, "layer": fc_layer}
        _initialize_layer_weights_according_to_activation_func(**kwargs)

        return None



    def forward(self, X):
        Y = self._fc_residual_block(X)
        Y = self._fc_layer(Y)

        return Y


_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._BasicResNetBuildingBlock
class _BasicResNetBuildingBlock(_cls_alias):
    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 max_kernel_size,
                 first_conv_layer_performs_downsampling,
                 final_activation_func,
                 mini_batch_norm_eps):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._BasicResNetBuildingBlock
        cls_alias.__init__(self, ctor_params)

        return None



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._BasicResNetStage
class _BasicResNetStage(_cls_alias):
    def __init__(self,
                 num_input_channels,
                 max_kernel_size,
                 num_building_blocks,
                 final_activation_func,
                 mini_batch_norm_eps):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        
        module_alias = emicroml.modelling._common
        cls_alias = module_alias._BasicResNetStage
        cls_alias.__init__(self, ctor_params)

        return None



class _SegmentationNet(torch.nn.Module):
    def __init__(self,
                 wavelet_name,
                 j_epsilon,
                 j_dashv,
                 num_pixels_across_each_cbed_pattern,
                 num_filters_in_first_conv_layer,
                 num_resnet_building_blocks_per_stage,
                 num_filters_per_bottleneck_conv_layer,
                 mini_batch_norm_eps):
        super().__init__()

        self._wavelet_name = \
            wavelet_name
        self._j_epsilon = \
            j_epsilon
        self._j_dashv = \
            j_dashv
        self._num_pixels_across_each_cbed_pattern = \
            num_pixels_across_each_cbed_pattern
        self._num_filters_in_first_conv_layer = \
            num_filters_in_first_conv_layer
        self._num_resnet_building_blocks_per_stage = \
            num_resnet_building_blocks_per_stage
        self._num_filters_per_bottleneck_conv_layer = \
            num_filters_per_bottleneck_conv_layer
        self._mini_batch_norm_eps = \
            mini_batch_norm_eps

        j_vdash = _get_j_vdash_from_wavelet_name(wavelet_name)
        self._j_vdash = j_vdash
        
        self._num_downsamplings = j_dashv-j_vdash+1
        self._idwt_block = _IDWTBlock(j_vdash)

        self._resnet_stages = self._generate_resnet_stages()
        self._downsampling_blocks = self._generate_downsampling_blocks()
        self._bottleneck_blocks = self._generate_bottleneck_blocks()
        self._prediction_blocks = self._generate_prediction_blocks()
        
        return None



    def _generate_resnet_stages(self):
        j_set = range(self._j_dashv, self._j_vdash-1, -1)
        
        kwargs = \
            {"num_input_channels": self._num_filters_in_first_conv_layer,
             "max_kernel_size": 3,
             "num_building_blocks": self._num_resnet_building_blocks_per_stage,
             "final_activation_func": torch.nn.ReLU(),
             "mini_batch_norm_eps": self._mini_batch_norm_eps}

        resnet_stages = tuple()
        for resnet_stage_idx, j in enumerate(j_set):
            resnet_stage = _BasicResNetStage(**kwargs)
            resnet_stages += (resnet_stage,)
            
            if resnet_stage_idx % 2 == 1:
                kwargs["num_input_channels"] *= 2
        
        resnet_stages = torch.nn.ModuleList(resnet_stages)

        return resnet_stages



    def _generate_downsampling_blocks(self):
        num_stages = len(self._resnet_stages)
        downsampling_block_indices = range(num_stages-1)

        kwargs = {"num_input_channels": self._num_filters_in_first_conv_layer,
                  "num_output_channels": self._num_filters_in_first_conv_layer,
                  "max_kernel_size": 3,
                  "first_conv_layer_performs_downsampling": True,
                  "final_activation_func": torch.nn.ReLU(),
                  "mini_batch_norm_eps": self._mini_batch_norm_eps}

        downsampling_blocks = tuple()
        for downsampling_block_idx in downsampling_block_indices:
            downsampling_block = _BasicResNetBuildingBlock(**kwargs)
            downsampling_blocks += (downsampling_block,)

            kwargs["num_input_channels"] = kwargs["num_output_channels"]
            if downsampling_block_idx % 2 == 0:
                kwargs["num_output_channels"] *= 2

        downsampling_blocks = torch.nn.ModuleList(downsampling_blocks)

        return downsampling_blocks



    def _generate_bottleneck_blocks(self):
        j_set = range(self._j_vdash,
                      self._j_epsilon+(self._j_vdash == self._j_epsilon))

        bottleneck_blocks = tuple()
        for j in j_set:
            resnet_stage_idx = -1 - (j-j_set[0])
            resnet_stage = self._resnet_stages[resnet_stage_idx]

            kwargs = {"num_pixels_across_each_cbed_pattern": \
                      self._num_pixels_across_each_cbed_pattern,
                      "num_input_channels": \
                      resnet_stage._num_ouput_channels,
                      "num_filters_per_conv_layer": \
                      self._num_filters_per_bottleneck_conv_layer,
                      "j_vdash": \
                      self._j_vdash,
                      "j": \
                      j,
                      "j_dashv": \
                      self._j_dashv,
                      "mini_batch_norm_eps": \
                      self._mini_batch_norm_eps}
            bottleneck_block = _BottleneckBlock(**kwargs)
            bottleneck_blocks = (bottleneck_block,) + bottleneck_blocks

        bottleneck_blocks = torch.nn.ModuleList(bottleneck_blocks)

        return bottleneck_blocks



    def _generate_prediction_blocks(self):
        j_set = (tuple(j
                       for j in range(self._j_epsilon-1, self._j_vdash-1, -1)
                       for _ in range(2))
                 + (self._j_vdash, self._j_vdash))

        prediction_blocks = tuple()
        for j in j_set:
            kwargs = {"j": j,
                      "j_dashv": self._j_dashv,
                      "mini_batch_norm_eps": self._mini_batch_norm_eps}
            prediction_block = _PredictionBlock(**kwargs)
            prediction_blocks += (prediction_block,)

        prediction_blocks = torch.nn.ModuleList(prediction_blocks)

        return prediction_blocks



    def forward(self, ml_inputs):
        dwt_coeffs = \
            self._predict_dwt_coeffs(ml_inputs)
        principal_disk_boundary_pt_sets = \
            self._perform_mra_reconstruction(dwt_coeffs)

        ml_predictions = dict()

        key = ("max_level_{}_approx_coeff_sets"
               "_of_principal_disk_boundary_pt_sets").format(self._wavelet_name)
        ml_predictions[key] = torch.stack(dwt_coeffs[:2], dim=2)

        key = "principal_disk_boundary_pt_sets"
        ml_predictions[key] = principal_disk_boundary_pt_sets

        return ml_predictions

        

    def _predict_dwt_coeffs(self, ml_inputs):
        dwt_coeffs = tuple()

        j_set = range(self._j_dashv-1, self._j_vdash-1, -1)
        prediction_block_count = 0

        Y_1 = self._get_and_enhance_cropped_cbed_pattern_images(ml_inputs)

        resnet_stage_idx = 0
        resnet_stage = self._resnet_stages[resnet_stage_idx]
        Y_2 = resnet_stage(Y_1)
        
        for j in j_set:
            downsampling_block_idx = \
                j-j_set[0]
            downsampling_block = \
                self._downsampling_blocks[downsampling_block_idx]
            Y_1 = \
                downsampling_block(Y_2)

            resnet_stage_idx = downsampling_block_idx+1
            resnet_stage = self._resnet_stages[resnet_stage_idx]
            Y_2 = resnet_stage(Y_1)

            if (j < self._j_epsilon) or (j == self._j_vdash):
                bottleneck_block_idx = (resnet_stage_idx
                                        - (self._j_dashv-(self._j_epsilon-1))
                                        + (self._j_vdash == self._j_epsilon))
                bottleneck_block = self._bottleneck_blocks[idx_2]
                Y_3 = bottleneck_block(Y_2)

                num_prediction_blocks_for_current_j = \
                    self._calc_num_prediction_blocks_for_current_j(j)

                start = prediction_block_count
                stop = start+num_prediction_blocks_for_current_j
                prediction_block_idx_subset = range(start, stop)

                for prediction_block_idx in prediction_block_idx_subset:
                    prediction_block = \
                        self._prediction_blocks[prediction_block_idx]
                    Y_4 = \
                        prediction_block(Y_3)
                    dwt_coeffs = \
                        (Y_4,) + dwt_coeffs
                    prediction_block_count += \
                        1

        return dwt_coeffs



    def _get_and_enhance_cropped_cbed_pattern_images(self, ml_inputs):
        kwargs = \
            {"image_stack": ml_inputs["cropped_cbed_pattern_images"]}
        enhanced_cropped_cbed_pattern_images = \
            _min_max_normalize_image_stack(**kwargs)

        gamma = 0.3

        enhanced_cropped_cbed_pattern_images = \
            torch.unsqueeze(enhanced_cropped_cbed_pattern_images, dim=1)
        enhanced_cropped_cbed_pattern_images = \
            torch.pow(enhanced_cropped_cbed_pattern_images, gamma)
        enhanced_cropped_cbed_pattern_images = \
            kornia.enhance.equalize(enhanced_cropped_cbed_pattern_images)

        kwargs = {"input": enhanced_cropped_cbed_pattern_images,
                  "min": 0,
                  "max": 1}
        enhanced_cropped_cbed_pattern_images = torch.clip(**kwargs)

        return enhanced_cropped_cbed_pattern_images



    def _calc_num_prediction_blocks_for_current_j(self, j):
        num_prediction_blocks_for_current_j = \
            2*(1 + (j == self._j_vdash)*(self._j_vdash != self._j_epsilon))

        return num_prediction_blocks_for_current_j
    


    def _perform_mra_reconstruction(self, dwt_coeffs)
        rescaled_approximation_coeffs = dwt_coeffs[:2]
        j_set = range(self._j_vdash, self._j_dashv)
        num_cartesian_cmpnts = 2
        
        for j in j_set:
            start = 2*((j-j_set[0])+1)
            stop = start+num_cartesian_cmpnts
            dwt_coeffs_idx_subset = range(start, stop)
            for dwt_coeffs_idx in dwt_coeffs_idx_subset:
                rescaled_a_alpha_j = rescaled_approximation_coeffs[-2]
                rescaled_d_alpha_j = (dwt_coeffs[dwt_coeffs_idx]
                                      if (j < self._j_epsilon)
                                      else 0*rescaled_a_alpha_j)
                rescaled_a_alpha_jP1 = self._idwt_block(rescaled_a_alpha_j,
                                                        rescaled_d_alpha_j)
                rescaled_approximation_coeffs += (rescaled_a_alpha_jP1,)

        principal_disk_boundary_pt_sets = \
            torch.stack(rescaled_approximation_coeffs[-2:], dim=2)

        return principal_disk_boundary_pt_sets



def _check_and_convert_wavelet_name(params):
    obj_name = "wavelet_name"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    wavelet_name = czekitout.convert.to_str_from_str_like(**kwargs)

    kwargs["accepted_strings"] = _generate_accepted_wavelet_names()
    czekitout.check.if_one_of_any_accepted_strings(**kwargs)

    return wavelet_name



def _check_and_convert_j_epsilon(params):
    obj_name = "j_epsilon"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}    
    j_epsilon = czekitout.convert.to_nonnegative_int(**kwargs)

    wavelet_name = _check_and_convert_wavelet_name(params)
    j_vdash = _get_j_vdash_from_wavelet_name(wavelet_name)

    current_func_name = "_check_and_convert_j_epsilon"

    if j_epsilon < j_vdash:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        args = (j_vdash,)
        err_msg = unformatted_err_msg.format(*args)
        raise ValueError(err_msg)

    return j_epsilon



def check_and_convert_j_dashv(params):
    obj_name = "j_dashv"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}    
    j_dashv = czekitout.convert.to_positive_int(**kwargs)

    j_epsilon = _check_and_convert_j_epsilon(params)

    current_func_name = "_check_and_convert_j_dashv"

    if j_dashv < j_epsilon:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return j_dashv



def _check_and_convert_num_pixels_across_each_cropped_cbed_pattern(params):
    obj_name = "num_pixels_across_each_cropped_cbed_pattern"

    func_alias = czekitout.convert.to_positive_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    num_pixels_across_each_cropped_cbed_pattern = func_alias(**kwargs)

    divisor = (_generate_divisor_3(params)
               if ("wavelet_name" in params)
               else _generate_divisor_4())

    current_func_name = ("_check_and_convert"
                         "_num_pixels_across_each_cropped_cbed_pattern")

    if num_pixels_across_each_cropped_cbed_pattern % divisor != 0:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(divisor)
        raise ValueError(err_msg)

    return num_pixels_across_each_cropped_cbed_pattern



def _generate_divisor_3(params):
    wavelet_name = _check_and_convert_wavelet_name(params)
    j_vdash = _get_j_vdash_from_wavelet_name(wavelet_name)

    j_dashv = _check_and_convert_j_dashv(params)
    
    divisor_3 = 2**(j_dashv-j_vdash)

    return divisor_3



def _generate_divisor_4():
    num_downsampling_steps_in_localization_net = \
        (emicroml.modelling._common._DistopticaNetEntryFlow._num_downsamplings
         + len(_building_block_counts_in_stages_of_localization_net))
    divisor_4 = \
        2**num_downsampling_steps_in_localization_net

    return divisor_4



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



def _get_device_name(device):
    kwargs = locals()
    module_alias = emicroml.modelling.cbed._common
    func_alias = module_alias._get_device_name
    device_name = func_alias(**kwargs)

    return device_name



_module_alias = \
    emicroml.modelling.cbed._common
_default_num_pixels_across_each_cropped_cbed_pattern = \
    _default_num_pixels_across_each_expected_cropping_window
_default_mini_batch_norm_eps = \
    _module_alias._default_mini_batch_norm_eps
_default_normalization_weights = \
    _module_alias._default_normalization_weights
_default_normalization_biases = \
    _module_alias._default_normalization_biases
_default_wavelet_name = \
    "db8"
_default_j_epsilon = \
    4
_default_j_dashv = \
    9
_default_normalizable_elems_of_ml_inputs_are_normalized = \
    _module_alias._default_normalizable_elems_of_ml_inputs_are_normalized
_default_unnormalize_normalizable_elems_of_ml_predictions = \
    _module_alias._default_unnormalize_normalizable_elems_of_ml_predictions



_module_alias = emicroml.modelling._common
_cls_alias = _module_alias._MLModel
class _MLModel(_cls_alias):
    def __init__(self,
                 num_pixels_across_each_cropped_cbed_pattern,
                 mini_batch_norm_eps,
                 normalization_weights,
                 normalization_biases,
                 ml_model_task,
                 wavelet_name,
                 j_epsilon,
                 j_dashv):
        current_cls_ctor_params = \
            {key: val
             for key, val in locals().items()
             if (key not in ("self", "__class__"))}
        base_cls_ctor_params = \
            self._get_base_cls_ctor_params(current_cls_ctor_params)

        kwargs = \
            {"num_pixels_across_each_cropped_cbed_pattern": \
             num_pixels_across_each_cropped_cbed_pattern}
        variable_axis_size_dict = \
            self._generate_variable_axis_size_dict(**kwargs)
        
        expected_keys_of_ml_inputs = \
            self._generate_expected_keys_of_ml_inputs()

        kwargs = {"for_constructing_ml_dataset": False,
                  "for_splitting_ml_dataset": False}
        ml_data_shape_analyzer = _MLDataShapeAnalyzer(**kwargs)

        module_alias = emicroml.modelling._common
        cls_alias = module_alias._MLModel
        kwargs = {"ml_data_normalizer": _generate_default_ml_data_normalizer(),
                  "ml_data_type_validator": _MLDataTypeValidator(),
                  "ml_data_value_validator": _MLDataValueValidator(),
                  "ml_data_shape_analyzer": ml_data_shape_analyzer,
                  "variable_axis_size_dict": variable_axis_size_dict,
                  "expected_keys_of_ml_inputs": expected_keys_of_ml_inputs,
                  "base_cls_ctor_params": base_cls_ctor_params}
        cls_alias.__init__(self, **kwargs)

        self._initialize_ml_model_cmpnts(ml_model_task)

        return None



    def _get_base_cls_ctor_params(self, current_cls_ctor_params):
        base_cls_ctor_params = current_cls_ctor_params.copy()

        ml_model_task = base_cls_ctor_params.pop(ml_model_task)

        if "localization" in ml_model_task:
            del base_cls_ctor_params["wavelet_name"]
            del base_cls_ctor_params["j_epsilon"]
            del base_cls_ctor_params["j_dashv"]

        return base_cls_ctor_params



    def _generate_variable_axis_size_dict(self,
                                          num_pixels_across_each_cropped_cbed_pattern):
        variable_axis_size_dict_keys = _generate_variable_axis_size_dict_keys()
        num_keys = len(variable_axis_size_dict_keys)

        variable_axis_size_dict = dict()
        for key_idx, key in enumerate(variable_axis_size_dict_keys):
            if key_idx == 0:
                variable_size_of_axis = None
            elif key_idx == num_keys-1:
                variable_size_of_axis = None
            else:
                variable_size_of_axis = num_pixels_across_each_cropped_cbed_pattern
                
            variable_axis_size_dict[key] = variable_size_of_axis

        return variable_axis_size_dict



    def _generate_expected_keys_of_ml_inputs(self):
        expected_keys_of_ml_inputs = ("cropped_cbed_pattern_images",)

        return expected_keys_of_ml_inputs



    def _check_and_convert_ctor_params(self, ctor_params):
        ctor_params = ctor_params.copy()

        global_symbol_table = globals()
        for ctor_param_name in ctor_params.keys():
            func_name = "_check_and_convert_" + ctor_param_name
            func_alias = global_symbol_table[func_name]
            ctor_params[ctor_param_name] = func_alias(params=ctor_params)

        return ctor_params



    def _initialize_ml_model_cmpnts(self, ml_model_task):
        core_nn_module_ctor_params = \
            {"num_pixels_across_each_cropped_cbed_pattern": \
             self._core_attrs["num_pixels_across_each_cropped_cbed_pattern"],
             "mini_batch_norm_eps": \
             self._core_attrs["mini_batch_norm_eps"]}

        if "localization" in ml_model_task:
            core_nn_module_cls = _LocalizationNet
        else:
            core_nn_module_cls = _SegmentationNet
            
            core_nn_module_ctor_params = \
                {**core_nn_module_ctor_params,
                 "wavelet_name": self._core_attrs["wavelet_name"],
                 "j_epsilon": self._core_attrs["j_epsilon"],
                 "j_dashv": self._core_attrs["j_dashv"],
                 "num_filters_in_first_conv_layer": 32,
                 "num_resnet_building_blocks_per_stage": 5,
                 "num_filters_per_bottleneck_conv_layer": 4}

        kwargs = core_nn_module_ctor_params
        self._core_nn_module = core_nn_module_cls(**kwargs)

        return None



    def forward(self, ml_inputs):
        ml_predictions = self._core_nn_module(ml_inputs)

        return ml_predictions



    def make_predictions(
            self,
            ml_inputs,
            unnormalize_normalizable_elems_of_ml_predictions=\
            _default_unnormalize_normalizable_elems_of_ml_predictions):
        kwargs = {"obj": ml_inputs, "obj_name": "ml_inputs"}
        ml_inputs = czekitout.convert.to_dict(**kwargs)

        params = \
            {"cropped_cbed_pattern_images": \
             ml_inputs.get("cropped_cbed_pattern_images", None),
             "name_of_obj_alias_of_cropped_cbed_pattern_images": \
             "ml_inputs['cropped_cbed_pattern_images']",
             "target_device": \
             next(self.parameters()).device}
        cropped_cbed_pattern_images = \
            _check_and_convert_cropped_cbed_pattern_images(params)
        ml_inputs["cbed_pattern_images"] = \
            cropped_cbed_pattern_images

        kwargs = {"ml_inputs": \
                  ml_inputs,
                  "unnormalize_normalizable_elems_of_ml_predictions": \
                  unnormalize_normalizable_elems_of_ml_predictions,
                  "normalizable_elems_of_ml_inputs_are_normalized": \
                  True}
        ml_predictions = super().make_predictions(**kwargs)

        return ml_predictions



    def _predict_distortion_models_via_cbed_disk_fit(
            self,
            cbed_pattern_images,
            cropping_window_centers=\
            _default_cropping_window_centers,
            auxiliary_distortion_estimation_model=\
            _default_auxiliary_distortion_estimation_model,
            auxiliary_localization_model=\
            _default_auxiliary_localization_model,
            cbed_disk_fitting_alg_params=\
            _default_disk_fitting_alg_params,
            distortion_model_sampling_grid_dims_in_pixels=\
            _default_distortion_model_sampling_grid_dims_in_pixels,
            distortion_model_least_squares_alg_params=\
            _default_distortion_model_least_squares_alg_params):
        # Perhaps ``cbed_disk_fitting_alg_params`` can store
        # ``cropping_window_centers``,
        # ``auxiliary_distortion_estimation_model``, and
        # ``auxiliary_localization_model``.

        return None



def _calc_mses_of_principal_disk_bounding_boxes(ml_predictions, ml_targets):
    calc_mse_loss = torch.nn.functional.mse_loss
    
    key = "principal_disk_bounding_boxes"

    kwargs = \
        {"input": ml_predictions[key],
         "target": ml_targets[key],
         "reduction": "none"}
    mses_of_principal_disk_bounding_boxes = \
        calc_mse_loss(**kwargs).mean(dim=(1,))

    return mses_of_principal_disk_bounding_boxes



def _calc_mads_of_principal_disk_bounding_boxes(ml_predictions, ml_targets):
    key = "principal_disk_bounding_boxes"

    mads_of_principal_disk_bounding_boxes = \
        (ml_predictions[key] - ml_targets[key]).abs().mean(dim=(1,))

    return mads_of_principal_disk_bounding_boxes



def _calc_meds_of_principal_disk_boundary_pt_sets(ml_predictions, ml_targets):
    calc_euclidean_distances = torch.linalg.vector_norm
    calc_eds = calc_euclidean_distances

    key = "principal_disk_boundary_pt_sets"

    kwargs = {"x": ml_predictions[key]-ml_targets[key], "dim": 2}
    meds_of_principal_disk_boundary_pt_sets = calc_eds(**kwargs).mean(dim=(1,))

    return meds_of_principal_disk_boundary_pt_sets



def _calc_meds_of_approx_coeff_sets(ml_predictions, ml_targets):
    calc_euclidean_distances = torch.linalg.vector_norm
    calc_eds = calc_euclidean_distances

    key = "principal_disk_boundary_pt_sets"
    j_dashv = round(np.log2(ml_predictions[key].shape[1]))

    for key in ml_predictions:
        if "approx_coeff_sets" in key:
            j_vdash = round(np.log2(ml_predictions[key].shape[1]))

            rescaling_factor = 2**((j_vdash-j_dashv)/2)
            
            kwargs = {"x": \
                      rescaling_factor*(ml_predictions[key]-ml_targets[key]),
                      "dim": \
                      2}
            meds_of_approx_coeff_sets = calc_eds(**kwargs).mean(dim=(1,))

    return meds_of_approx_coeff_sets



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

        global_symbol_table = globals()

        for key_1 in ml_predictions:
            partial_key_set_1 = (("mses", "mads")
                                 if ("boxes" in key_1)
                                 else ("meds",))
            
            for partial_key_1 in partial_key_set_1:
                partial_key_2 = ("approx_coeff_sets"
                                 if ("approx_coeff_sets" in key_1)
                                 else key_1)

                key_2 = "{}_of_{}".format(partial_key_1, key_1)
                
                func_name = "_calc_{}_of_{}".format(partial_key_1,
                                                    partial_key_2)
                func_alias = global_symbol_table[func_name]

                kwargs = {"ml_predictions": ml_predictions,
                          "ml_targets": ml_targets}
                metrics_of_current_mini_batch[key_2] = func_alias(**kwargs)

        return metrics_of_current_mini_batch



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

        key_set_1 = tuple(key_1
                          for key_1
                          in metrics_of_current_mini_batch
                          if "mads" not in key_1)

        losses_of_current_mini_batch = {"total": 0.0}

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

_check_and_convert_num_pixels_across_each_expected_cropping_window_err_msg_1 = \
    ("The object ``num_pixels_across_each_expected_cropping_window`` must be "
     "positive integer that is divisible {}.")

_check_and_convert_num_pixels_across_each_cropping_window_err_msg_1 = \
    ("The object ``num_pixels_across_each_cropping_window`` must be positive "
     "integer that is divisible {}.")

_default_cropped_cbed_pattern_generator_err_msg_1 = \
    ("The principal CBED disk of the cropped CBED pattern must not be clipped "
     "nor overlapping with any other CBED disks.")
_default_cropped_cbed_pattern_generator_err_msg_2 = \
    ("The cropped CBED pattern generator{} has exceeded its programmed maximum "
     "number of attempts{} to generate a valid cropped CBED pattern: see "
     "traceback for details.")

_generate_cropped_cbed_pattern_signal_err_msg_1 = \
    _default_cropped_cbed_pattern_generator_err_msg_2
_generate_cropped_cbed_pattern_signal_err_msg_2 = \
    ("The object ``cropped_cbed_pattern_generator`` did not generate a valid "
     "cropped CBED pattern, i.e. it did not generate an object of the type "
     "``fakecbed.discretized.CroppedCBEDPattern``.")

_check_cropped_cbed_pattern_signal_err_msg_1 = \
    ("The object ``cropped_cbed_pattern_generator`` must generate a cropped "
     "CBED pattern image that contains a principal CBED disk that is not "
     "clipped nor overlapping with any other CBED disks.")
_check_cropped_cbed_pattern_signal_err_msg_2 = \
    ("The object ``cropped_cbed_pattern_generator`` must generate a cropped "
     "CBED pattern image with the horizontal dimension, in units of pixels, "
     "being equal to the vertical dimension, in units of pixels.")
_check_cropped_cbed_pattern_signal_err_msg_3 = \
    ("The object ``cropped_cbed_pattern_generator`` must generate a cropped "
     "CBED pattern image with the horizontal dimension, in units of pixels, "
     "being equal to a positive integer that is divisible by {}.")

_unnormalized_ml_data_instance_generator_err_msg_1 = \
    ("The object ``cropped_cbed_pattern_generator`` must generate cropped CBED "
     "patterns of consistent dimensions.")
_unnormalized_ml_data_instance_generator_err_msg_2 = \
    ("The object ``cropped_cbed_pattern_generator`` must generate cropped CBED "
     "patterns, where for each pattern, the same number of points on the "
     "boundary of the principal CBED disk are sampled.")

_custom_value_checker_for_cropped_cbed_pattern_images_err_msg_1 = \
    ("The HDF5 dataset at the HDF5 path ``'{}'`` of the HDF5 file at the file "
     "path ``'{}'`` must contain only images that are normalized such that the "
     "minimum and maximum pixel values are equal to zero and unity "
     "respectively for each image.")
_custom_value_checker_for_cropped_cbed_pattern_images_err_msg_2 = \
    ("The object ``{}['{}']`` must contain only images that are normalized "
     "such that the minimum and maximum pixel values are equal to zero and "
     "unity respectively for each image.")

_ml_data_shape_analyzer_err_msg_1 = \
    ("The HDF5 dataset ``hdf5_dataset``at the HDF5 path ``'{}'`` of the HDF5 "
     "file at the file path ``'{}'`` must satisfy "
     "``hdf5_dataset.shape[1] >= {}`` and ``hdf5_dataset.shape[1]`` must be a "
     "power of two.")

_check_and_convert_bounding_box_marker_style_kwargs_err_msg_1 = \
    ("The object ``bounding_box_marker_style_kwargs`` must either be set to "
     "``None`` or a dictionary of valid keyword arguments for the constructor "
     "of the class `hyperspy.api.plot.markers.Rectangles` that specify "
     "stylistic properties of markers represented by the aforementioned class, "
     "i.e. keyword arguments other than ``'offset_transform'``, "
     "``'transform'``, ``'shift'``, ``'plot_on_signal'``, ``'name'``, "
     "``'ScalarMappable_array'``, ``'offsets'``, ``'heights'``, ``'widths'``, "
     "and ``'angles'``.")

_check_and_convert_boundary_pt_marker_style_kwargs_err_msg_1 = \
    ("The object ``boundary_pt_marker_style_kwargs`` must either be set to "
     "``None`` or a dictionary of valid keyword arguments for the constructor "
     "of the class `hyperspy.api.plot.markers.Points` that specify stylistic "
     "properties of markers represented by the aforementioned class, i.e. "
     "keyword arguments other than ``'offset_transform'``, ``'transform'``, "
     "``'shift'``, ``'plot_on_signal'``, ``'name'``, "
     "``'ScalarMappable_array'``, ``'offsets'``, and ``'sizes'``. "
     "Additionally, the dictionary can contain a `slice` object stored in a "
     "dictionary item with the key ``'single_dim_slice'``.")

_check_and_convert_cropped_cbed_pattern_images_err_msg_1 = \
    ("The object ``{}`` must be an array of three dimensions.")

_check_and_convert_j_epsilon_err_msg_1 = \
    ("The object ``j_epsilon`` must be a nonnegative integer that is greater "
     "than or equal to ``np.ceil(np.log2(2*int(wavelet_name[2:])-1))``, i.e. "
     "``{}``, where ``np`` is an alias for the NumPy library ``numpy``.")

_check_and_convert_j_dashv_err_msg_1 = \
    ("The object ``j_dashv`` must be a positive integer that is greater than "
     "or equal to ``j_epsilon``.")

_check_and_convert_num_pixels_across_each_cropped_cbed_pattern_err_msg_1 = \
    ("The object ``num_pixels_across_each_cropped_cbed_pattern`` must be a "
     "positive integer that is divisible by ``{}``, to be in accordance with "
     "``wavelet_name`` and ``j_dashv``.")
