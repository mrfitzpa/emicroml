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
r"""A script that is called by various other scripts used for generating
individual machine learning (ML) datasets that can be used to test ML models for
a specified task. The script retrieves an image, presumed to be an simulated
undistorted CBED pattern, and uses it to generate a set of distorted "fake" CBED
patterns that are then stored in a new ML dataset.

The correct form of the command to run the script is::

  python execute_main_action_steps.py \
         --ml_model_task=<ml_model_task> \
         --disk_size_idx=<disk_size_idx> \
         --disk_size=<disk_size> \
         --ml_dataset_idx=<ml_dataset_idx> \
         --data_dir_1=<data_dir_1> \
         --data_dir_2=<data_dir_2>

where ``<ml_model_task>`` is one of a set of accepted strings that specifies the
ML model task, ``<disk_size_idx>`` is an integer that is used to select a seed
for random number generation, ``<disk_size>`` is one of a set of accepted
strings that describes the size of the CBED disks in the distorted CBED patterns
to be stored in the ML dataset to be generated; ``<ml_dataset_idx>`` is an
integer that is used to label the ML dataset; ``<data_dir_1>`` is the absolute
path to an existing directory or one to be created, within which the output data
is to be saved.; and ``<data_dir_2>`` is the absolute path to a directory
directly containing the file that stores the undistorted CBED pattern to use to
generate the distorted CBED patterns that will make up the individual ML dataset
to be generated.

At the moment, the only accepted value of ``<ml_model_task>`` is
``cbed/distortion/estimation``, which specifies that the ML model task is
distortion estimation in CBED. ``<disk_size_idx>`` and ``<ml_dataset_idx>`` can
be any nonnegative integers. The accepted values of ``<disk_size>`` are
``small``, ``medium``, and ``large``. ``<data_dir_1>`` can be any valid absolute
path to any valid existing directory or one to be created. ``<data_dir_2>`` must
be the absolute path to an existing directory that contains the output files
resulting from an execution of the function :func:`prismatique.stem.sim.run`
that stores at least one CBED intensity pattern.

The only non-temporary output data generated from this script is a single HDF5
file, which stores the ML dataset. Upon successful execution of the script, the
HDF5 file is saved to
``<data_dir_1>/ml_datasets/ml_datasets_for_ml_model_test_set_1/ml_datasets_with_cbed_patterns_of_MoS2_on_amorphous_C/ml_datasets_with_<disk_size>_sized_disks/ml_dataset_<ml_dataset_idx>.h5``.

This script uses the module
:mod:`emicroml.modelling.cbed.distortion.estimation`. It is recommended that you
consult the documentation of said module as you explore the remainder of this
script. Furthermore, this script also uses the packages :mod:`embeam`,
:mod:`fakecbed`, :mod:`distoptica`, and :mod:`prismatique` It is recommended
that you consult the documentation of these packages as well.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For accessing imported modules via their names stored as strings.
import sys



# For general array handling.
import numpy as np
import torch

# For calculating electron beam wavelengths given mean beam energies.
import embeam

# For generating distortion models.
import distoptica

# For generating fake CBED patterns.
import fakecbed

# For loading STEM simulation parameters.
import prismatique

# For inpainting images.
import skimage.restoration



# For generating ML datsets.
import emicroml.modelling.cbed.distortion.estimation
import emicroml.modelling.cbed.disk.localization
import emicroml.modelling.cbed.disk.segmentation



##############################################
## Define classes, functions, and constants ##
##############################################

class CBEDPatternGenerator():
    def __init__(self,
                 ml_model_task,
                 path_to_stem_multislice_sim_intensity_output,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 device_name,
                 path_to_stem_multislice_sim_params):
        self._ml_model_task = \
            ml_model_task
        self._path_to_stem_multislice_sim_intensity_output = \
            path_to_stem_multislice_sim_intensity_output
        self._max_num_disks_in_any_cbed_pattern = \
            max_num_disks_in_any_cbed_pattern
        self._rng_seed = \
            rng_seed
        self._device_name = \
            device_name

        self.max_num_disks_in_any_cbed_pattern = \
            max_num_disks_in_any_cbed_pattern
        
        kwargs = {"path_to_stem_multislice_sim_params": \
                  path_to_stem_multislice_sim_params}
        self._store_relevant_stem_multislice_sim_params(**kwargs)
        
        self._wavelength = embeam.wavelength(self._mean_beam_energy)
        
        self._store_stem_multislice_sim_intensity_pattern_signal()
        self._store_property_subset_of_stem_multislice_sim_intensity_output()

        undistorted_image_dims_in_pixels = \
            self._stem_multislice_sim_intensity_pattern_signal.data.shape[-2:]
        self._sampling_grid_dims_in_pixels = \
            (undistorted_image_dims_in_pixels[0]//2,
             undistorted_image_dims_in_pixels[1]//2)

        self._ml_model_task_module = self._get_ml_model_task_module()
        
        self._distortion_model_generator = \
            self._generate_distortion_model_generator()
        self._undistorted_tds_model = \
            self._generate_undistorted_tds_model()
        self._undistorted_disks = \
            self._generate_undistorted_disks()

        self._rng = np.random.default_rng(self._rng_seed)

        self._initialize_and_cache_cbed_pattern_params()

        return None



    def _store_relevant_stem_multislice_sim_params(
            self, path_to_stem_multislice_sim_params):
        kwargs = {"filename": path_to_stem_multislice_sim_params,
                  "skip_validation_and_conversion": True}
        stem_multislice_sim_params = prismatique.stem.sim.Params.load(**kwargs)
        
        stem_system_model_params = \
            stem_multislice_sim_params.core_attrs["stem_system_model_params"]
        probe_model_params = \
            stem_system_model_params.core_attrs["probe_model_params"]
        gun_model_params = \
            probe_model_params.core_attrs["gun_model_params"]

        self._convergence_semiangle = \
            probe_model_params.core_attrs["convergence_semiangle"]  # In mrads.
        self._mean_beam_energy = \
            gun_model_params.core_attrs["mean_beam_energy"]  # in keVs.

        return None



    def _store_stem_multislice_sim_intensity_pattern_signal(self):
        kwargs = {"filename": \
                  self._path_to_stem_multislice_sim_intensity_output,
                  "multi_dim_slice": \
                  (0, 0)}
        signal, _ = prismatique.load.cbed_intensity_patterns(**kwargs)
        self._stem_multislice_sim_intensity_pattern_signal = signal

        return None



    def _store_property_subset_of_stem_multislice_sim_intensity_output(self):
        signal = self._stem_multislice_sim_intensity_pattern_signal

        k_x_axis_label = "$k_x$"
        self._k_x_offset = signal.axes_manager[k_x_axis_label].offset
        self._k_x_scale = signal.axes_manager[k_x_axis_label].scale
        self._k_x_size = signal.axes_manager[k_x_axis_label].size

        k_y_axis_label = "$k_y$"
        self._k_y_offset = signal.axes_manager[k_y_axis_label].offset
        self._k_y_scale = signal.axes_manager[k_y_axis_label].scale
        self._k_y_size = signal.axes_manager[k_y_axis_label].size

        return None



    def _get_ml_model_task_module(self):
        ml_model_task = self._ml_model_task

        module_name = \
            "emicroml.modelling.{}".format(ml_model_task).replace("/", ".")
        ml_model_task_module = \
            sys.modules[module_name]

        return ml_model_task_module



    def _generate_distortion_model_generator(self):
        ml_model_task_module = self._ml_model_task_module

        cls_alias = ml_model_task_module.DefaultDistortionModelGenerator
        kwargs = {"reference_pt": \
                  (0.5, 0.5),
                  "rng_seed": \
                  self._rng_seed,
                  "sampling_grid_dims_in_pixels": \
                  self._sampling_grid_dims_in_pixels,
                  "least_squares_alg_params": \
                  None,
                  "device_name": \
                  self._device_name}
        distortion_model_generator = cls_alias(**kwargs)

        return distortion_model_generator



    def _generate_undistorted_tds_model(self):
        reference_pt_of_distortion_model_generator = \
            self._distortion_model_generator.core_attrs["reference_pt"]

        kwargs = {"center": reference_pt_of_distortion_model_generator,
                  "widths": 4*(1,),
                  "rotation_angle": 0,
                  "val_at_center": 0,
                  "functional_form": "asymmetric_gaussian"}
        tds_peak = fakecbed.shapes.Peak(**kwargs)

        kwargs = {"peaks": (tds_peak,), "constant_bg": 0}
        undistorted_tds_model = fakecbed.tds.Model(**kwargs)

        return undistorted_tds_model



    def _generate_undistorted_disks(self):
        a = 3.1604  # "a" lattice parameter of MoS2 in Å.

        # Magnitude of either primitive reciprocal lattice vector of MoS2.
        b_1_mag = (2*np.pi) * (2/a/np.sqrt(3))

        # MoS2 rescaled (non-primitive) unit-cell reciprocal lattice vectors.
        q_1 = (b_1_mag / (2*np.pi)) * np.array([np.sqrt(3), 0.0])
        q_2 = (b_1_mag / (2*np.pi)) * np.array([0.0, 1.0])

        # Positions of disks in unit cell.
        delta_disk_1 = (0/2)*q_1 + (0/2)*q_2
        delta_disk_2 = (1/2)*q_1 + (1/2)*q_2

        # Disk unit cell.
        disk_unit_cell = np.array((delta_disk_1, delta_disk_2))

        # Determine the number of tiles of the disk unit cell.
        k_x_tiling_indices = self._calc_k_x_tiling_indices(q_1)
        k_y_tiling_indices = self._calc_k_y_tiling_indices(q_2)

        undistorted_disks = tuple()

        for k_x_tiling_idx in k_x_tiling_indices:
            for k_y_tiling_idx in k_y_tiling_indices:
                shift = k_x_tiling_idx*q_1 + k_y_tiling_idx*q_2
                current_disk_cell = np.array(tuple(delta_disk+shift
                                                   for delta_disk
                                                   in disk_unit_cell))

                for (k_x_c_support, k_y_c_support) in current_disk_cell:
                    kwargs = {"k_x_c_support": k_x_c_support,
                              "k_y_c_support": k_y_c_support}
                    undistorted_disk = self._generate_undistorted_disk(**kwargs)
                    undistorted_disks += (undistorted_disk,)

        return undistorted_disks



    def _calc_k_x_tiling_indices(self, q_1):
        k_R_support = self._calc_k_R_support()

        q_1_norm = np.linalg.norm(q_1)

        k_x_offset = self._k_x_offset
        k_x_scale = self._k_x_scale
        k_x_size = self._k_x_size

        k_x_max_candidate_1 = (abs(k_x_offset)
                               + k_R_support)
        k_x_max_candidate_2 = (abs(k_x_offset + k_x_scale*(k_x_size-1))
                               + k_R_support)
        
        k_x_max = max(k_x_max_candidate_1, k_x_max_candidate_2)

        max_k_x_tiling_idx = int(k_x_max // q_1_norm)
        min_k_x_tiling_idx = -max_k_x_tiling_idx

        k_x_tiling_indices = range(min_k_x_tiling_idx, max_k_x_tiling_idx+1)

        return k_x_tiling_indices



    def _calc_k_R_support(self):
        k_R_support = (self._convergence_semiangle/1000) / self._wavelength

        return k_R_support



    def _generate_undistorted_disk(self, k_x_c_support, k_y_c_support):
        kwargs = {"k_x_coord": k_x_c_support}
        u_x_c_support = self._k_x_coord_to_u_x_coord(**kwargs)

        kwargs = {"k_y_coord": k_y_c_support}
        u_y_c_support = self._k_y_coord_to_u_y_coord(**kwargs)

        kwargs = {"center": (u_x_c_support, u_y_c_support),
                  "radius": self._calc_u_R_support(),
                  "intra_shape_val": 1,
                  "skip_validation_and_conversion": True}
        undistorted_disk_support = fakecbed.shapes.Circle(**kwargs)

        kwargs = {"support": undistorted_disk_support,
                  "intra_support_shapes": tuple(),
                  "skip_validation_and_conversion": True}
        undistorted_disk = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

        return undistorted_disk



    def _calc_k_y_tiling_indices(self, q_2):
        k_R_support = self._calc_k_R_support()

        q_2_norm = np.linalg.norm(q_2)

        k_y_offset = self._k_y_offset
        k_y_scale = self._k_y_scale
        k_y_size = self._k_y_size

        k_y_max_candidate_1 = (abs(k_y_offset)
                               + k_R_support)
        k_y_max_candidate_2 = (abs(k_y_offset + k_y_scale*(k_y_size-1))
                               + k_R_support)
        
        k_y_max = max(k_y_max_candidate_1, k_y_max_candidate_2)

        max_k_y_tiling_idx = int(k_y_max // q_2_norm)
        min_k_y_tiling_idx = -max_k_y_tiling_idx

        k_y_tiling_indices = range(min_k_y_tiling_idx, max_k_y_tiling_idx+1)

        return k_y_tiling_indices



    def _k_x_coord_to_u_x_coord(self, k_x_coord):
        u_x_offset = 0.5/(self._k_x_size/2)
        u_x_scale = 1/(self._k_x_size/2)

        n_x = self._k_x_coord_to_pixel_coord(k_x_coord)

        u_x_coord = u_x_offset + n_x*u_x_scale

        return u_x_coord



    def _k_x_coord_to_pixel_coord(self, k_x_coord):
        pixel_coord = (k_x_coord-(self._k_x_offset/2))/self._k_x_scale
        
        return pixel_coord



    def _k_y_coord_to_u_y_coord(self, k_y_coord):
        u_y_offset = 1-(1-0.5)/(self._k_y_size/2)
        u_y_scale = -1/(self._k_y_size/2)

        n_y = self._k_y_coord_to_pixel_coord(k_y_coord)

        u_y_coord = u_y_offset + n_y*u_y_scale

        return u_y_coord



    def _k_y_coord_to_pixel_coord(self, k_y_coord):
        pixel_coord = (k_y_coord-(self._k_y_offset/2))/self._k_y_scale

        return pixel_coord



    def _calc_u_R_support(self):
        k_R_support = self._calc_k_R_support()

        u_x_scale = 1/(self._k_x_size/2)
        u_R_support = (k_R_support/self._k_x_scale)*u_x_scale

        return u_R_support



    def _initialize_and_cache_cbed_pattern_params(self):
        self._cbed_pattern_params = \
            {"undistorted_tds_model": \
             self._undistorted_tds_model,
             "undistorted_disks": \
             self._undistorted_disks,
             "undistorted_misc_shapes": \
             tuple(),
             "undistorted_outer_illumination_shape": \
             None,
             "gaussian_filter_std_dev": \
             0,
             "num_pixels_across_pattern": \
             self._sampling_grid_dims_in_pixels[0],
             "distortion_model": \
             None,
             "apply_shot_noise": \
             False,
             "rng_seed": \
             None,
             "cold_pixels": \
             tuple(),
             "detector_partition_width_in_pixels": \
             4,
             "mask_frame": \
             (0, 0, 0, 0)}

        return None



    def generate(self):
        cbed_pattern_params = self._cbed_pattern_params

        generation_attempt_count = 0
        max_num_generation_attempts = 10
        cbed_pattern_generation_has_not_been_completed = True

        while cbed_pattern_generation_has_not_been_completed:
            try:
                cbed_pattern_params["distortion_model"] = \
                    self._distortion_model_generator.generate()

                key_subset = ("mask_frame",
                              "undistorted_outer_illumination_shape")
                for key in key_subset:
                    method_name = "_generate_{}".format(key)
                    method_alias = getattr(self, method_name)
                    kwargs = {"cbed_pattern_params": cbed_pattern_params}
                    cbed_pattern_params[key] = method_alias(**kwargs)

                kwargs = cbed_pattern_params
                cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

                disk_clipping_registry = \
                    cbed_pattern.get_disk_clipping_registry(deep_copy=False)
                num_non_clipped_disks = \
                    (~disk_clipping_registry).sum().item()

                min_num_disks_in_any_cbed_pattern = 4

                if num_non_clipped_disks < min_num_disks_in_any_cbed_pattern:
                    unformatted_err_msg = _cbed_pattern_generator_err_msg_1
                    args = (min_num_disks_in_any_cbed_pattern,)
                    err_msg = unformatted_err_msg.format(*args)
                    raise ValueError(err_msg)

                cbed_pattern_generation_has_not_been_completed = False
            except:
                generation_attempt_count += 1                
                if generation_attempt_count == max_num_generation_attempts:
                    unformatted_err_msg = _cbed_pattern_generator_err_msg_2
                    args = ("", " ({})".format(max_num_generation_attempts))
                    err_msg = unformatted_err_msg.format(*args)                    
                    raise RuntimeError(err_msg)

        kwargs = {"overriding_image": self._generate_overriding_image()}
        cbed_pattern.override_image_then_reapply_mask(**kwargs)

        return cbed_pattern



    def _generate_mask_frame(self, cbed_pattern_params):
        distortion_model = cbed_pattern_params["distortion_model"]

        min_fractional_mask_frame_width = \
            self._distortion_model_generator._min_fractional_mask_frame_width
        max_fractional_mask_frame_width = \
            self._distortion_model_generator._max_fractional_mask_frame_width

        sampling_grid_dims_in_pixels = \
            self._sampling_grid_dims_in_pixels
        num_pixels_across_each_cbed_pattern = \
            sampling_grid_dims_in_pixels[0]

        attr_name = "mask_frame_of_distorted_then_resampled_images"
        quadruple_1 = np.array(getattr(distortion_model, attr_name),
                               dtype=float)
        quadruple_1[:2] /= sampling_grid_dims_in_pixels[0]
        quadruple_1[2:] /= sampling_grid_dims_in_pixels[1]

        kwargs = {"low": min_fractional_mask_frame_width,
                  "high": max_fractional_mask_frame_width,
                  "size": 4}
        quadruple_2 = self._rng.uniform(**kwargs)

        trivial_mask_frame_is_not_to_be_generated = \
            (self._rng.choice((True, False), p=(1/2, 1-1/2)).item()
             * ("cbed/disk" not in self._ml_model_task))

        mask_frame = \
            tuple(np.round(((quadruple_1>=quadruple_2)*quadruple_1
                            + (quadruple_1<quadruple_2)*quadruple_2)
                           * num_pixels_across_each_cbed_pattern).astype(int)
                  * trivial_mask_frame_is_not_to_be_generated)

        return mask_frame

    

    def _generate_undistorted_outer_illumination_shape(self,
                                                       cbed_pattern_params):
        mask_frame = cbed_pattern_params["mask_frame"]

        undistorted_outer_illumination_shape_is_elliptical = \
            self._rng.choice((True, False), p=(3/4, 1/4)).item()

        if undistorted_outer_illumination_shape_is_elliptical:
            method_name = ("_generate_elliptical"
                           "_undistorted_outer_illumination_shape")
        else:
            method_name = ("_generate_generic"
                           "_undistorted_outer_illumination_shape")
            
        method_alias = getattr(self, method_name)
        kwargs = {"mask_frame": mask_frame}
        undistorted_outer_illumination_shape = method_alias(**kwargs)

        return undistorted_outer_illumination_shape



    def _generate_elliptical_undistorted_outer_illumination_shape(self,
                                                                  mask_frame):
        rng = self._rng

        u_r_E = abs(rng.normal(loc=0, scale=1/20))
        u_phi_E = rng.uniform(low=0, high=2*np.pi)
        cos = np.cos
        sin = np.sin

        reference_pt_of_distortion_model_generator = \
            self._distortion_model_generator.core_attrs["reference_pt"]

        center = (reference_pt_of_distortion_model_generator[0]
                  - u_r_E*cos(u_phi_E),
                  reference_pt_of_distortion_model_generator[1]
                  - u_r_E*sin(u_phi_E))
        center = (center[0].item(), center[1].item())

        kwargs = {"reference_pt_of_distortion_model_generator": \
                  reference_pt_of_distortion_model_generator,
                  "mask_frame": \
                  mask_frame}
        semi_major_axis = self._generate_semi_major_axis(**kwargs)

        kwargs = {"center": center,
                  "semi_major_axis": semi_major_axis,
                  "eccentricity": abs(rng.uniform(low=0, high=0.6)),
                  "rotation_angle": rng.uniform(low=0, high=2*np.pi),
                  "intra_shape_val": 1.0,
                  "skip_validation_and_conversion": True}
        undistorted_outer_illumination_shape = fakecbed.shapes.Ellipse(**kwargs)

        return undistorted_outer_illumination_shape



    def _generate_semi_major_axis(self,
                                  reference_pt_of_distortion_model_generator,
                                  mask_frame):
        rng = self._rng

        choices = ((sum(mask_frame) != 0)*1e6, 1e6)

        loc = (max(reference_pt_of_distortion_model_generator[0],
                   1-reference_pt_of_distortion_model_generator[0],
                   reference_pt_of_distortion_model_generator[1],
                   1-reference_pt_of_distortion_model_generator[1])
               + rng.choice(choices, p=(3/4, 1-3/4)).item())
        semi_major_axis = loc + rng.uniform(low=-loc/6, high=loc/6)

        return semi_major_axis



    def _generate_generic_undistorted_outer_illumination_shape(self,
                                                               mask_frame):
        rng = self._rng

        u_r_GB = abs(rng.normal(loc=0, scale=1/20))
        u_phi_GB = rng.uniform(low=0, high=2*np.pi)
        cos = np.cos
        sin = np.sin

        reference_pt_of_distortion_model_generator = \
            self._distortion_model_generator.core_attrs["reference_pt"]

        radial_reference_pt_of_blob = \
            (reference_pt_of_distortion_model_generator[0]
             - u_r_GB*cos(u_phi_GB).item(),
             reference_pt_of_distortion_model_generator[1]
             - u_r_GB*sin(u_phi_GB).item())

        kwargs = {"reference_pt_of_distortion_model_generator": \
                  reference_pt_of_distortion_model_generator,
                  "mask_frame": \
                  mask_frame}
        radial_amplitude = self._generate_radial_amplitude(**kwargs)

        num_amplitudes = rng.integers(low=2, high=4, endpoint=True).item()
        radial_amplitudes = (radial_amplitude,)
        radial_phases = tuple()
        for amplitude_idx in range(1, num_amplitudes+1):
            kwargs = {"low": 0,
                      "high": radial_amplitudes[0]/num_amplitudes/4}
            radial_amplitude = rng.uniform(**kwargs)
            radial_amplitudes += (radial_amplitude,)

            kwargs = {"low": 0,
                      "high": 2*np.pi/amplitude_idx}
            radial_phase = rng.uniform(**kwargs)
            radial_phases += (radial_phase,)

        kwargs = \
            {"radial_reference_pt": radial_reference_pt_of_blob,
             "radial_amplitudes": radial_amplitudes,
             "radial_phases": radial_phases,
             "intra_shape_val": 1.0,
             "skip_validation_and_conversion": True}
        undistorted_outer_illumination_shape = \
            fakecbed.shapes.GenericBlob(**kwargs)

        return undistorted_outer_illumination_shape



    def _generate_radial_amplitude(self,
                                   reference_pt_of_distortion_model_generator,
                                   mask_frame):
        kwargs = {"reference_pt_of_distortion_model_generator": \
                  reference_pt_of_distortion_model_generator,
                  "mask_frame": \
                  mask_frame}
        semi_major_axis = self._generate_semi_major_axis(**kwargs)
        radial_amplitude = semi_major_axis

        return radial_amplitude



    def _generate_overriding_image(self):
        distortion_model = self._cbed_pattern_params["distortion_model"]
        
        device = distortion_model.device
        signal = self._stem_multislice_sim_intensity_pattern_signal

        input_tensor_to_sample = torch.from_numpy(signal.data)
        input_tensor_to_sample = input_tensor_to_sample.to(device)
        input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)
        input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)

        method_name = "get_sampling_grid"
        method_alias = getattr(distortion_model, method_name)
        sampling_grid = method_alias(deep_copy=False)

        method_name = "get_flow_field_of_coord_transform_right_inverse"
        method_alias = getattr(distortion_model, method_name)
        flow_field = method_alias(deep_copy=False)

        grid_shape = (1,) + self._sampling_grid_dims_in_pixels + (2,)
        grid = torch.zeros(grid_shape,
                           dtype=input_tensor_to_sample.dtype,
                           device=device)
        grid[0, :, :, 0] = flow_field[0]+sampling_grid[0]-0.5
        grid[0, :, :, 1] = -(flow_field[1]+sampling_grid[1]-0.5)

        kwargs = {"input": input_tensor_to_sample,
                  "grid": grid,
                  "mode": "bilinear",
                  "padding_mode": "zeros",
                  "align_corners": False}
        overriding_image = torch.nn.functional.grid_sample(**kwargs)[0, 0]

        kwargs = {"input_image": overriding_image}
        overriding_image = self._apply_shot_noise_to_image(**kwargs)
        overriding_image = self._apply_detector_partition_inpainting(**kwargs)

        return overriding_image



    def _apply_shot_noise_to_image(self, input_image):
        torch_rng_seed = self._rng.integers(low=0, high=2**32-1).item()
        torch_rng = torch.Generator(device=input_image.device)
        torch_rng = torch_rng.manual_seed(torch_rng_seed)
        output_image = torch.poisson(input_image, torch_rng)

        return output_image



    def _apply_detector_partition_inpainting(self, input_image):
        N_DPW = self._cbed_pattern_params["detector_partition_width_in_pixels"]

        k_I_1 = ((input_image.shape[1]-1)//2) - (N_DPW//2)
        k_I_2 = k_I_1 + N_DPW - 1

        inpainting_mask = np.zeros(input_image.shape, dtype=bool)
        inpainting_mask[k_I_1:k_I_2+1, :] = True
        inpainting_mask[:, k_I_1:k_I_2+1] = True

        kwargs = {"image": input_image.numpy(force=True),
                  "mask": inpainting_mask}
        output_image = skimage.restoration.inpaint_biharmonic(**kwargs)
        output_image = torch.from_numpy(output_image)
        output_image = output_image.to(device=input_image.device,
                                       dtype=input_image.dtype)

        return output_image



class CroppedCBEDPatternGenerator():
    def __init__(self,
                 ml_model_task,
                 path_to_stem_multislice_sim_intensity_output,
                 max_num_disks_in_any_cbed_pattern,
                 rng_seed,
                 device_name,
                 path_to_stem_multislice_sim_params,
                 path_to_ml_training_dataset,
                 num_pixels_across_each_cropping_window):
        kwargs = {key: val
                  for key, val in locals().items()
                  if (key not in ("self", "__class__"))}
        del kwargs["path_to_ml_training_dataset"]
        del kwargs["num_pixels_across_each_cropping_window"]
        self._cbed_pattern_generator = CBEDPatternGenerator(**kwargs)

        self._path_to_ml_training_dataset = \
            path_to_ml_training_dataset

        self._num_pixels_across_each_cropping_window = \
            num_pixels_across_each_cropping_window

        self._principal_disk_candidate_registry = \
            self._generate_principal_disk_candidate_registry()

        self._rng = self._cbed_pattern_generator._rng

        self._initialize_and_cache_cbed_pattern_params()

        N_dot = self._cropped_cbed_pattern_params["disk_boundary_sample_size"]

        self.resolution_level_of_disk_boundary_sample_size = \
            round(np.round(np.log2(N_dot)))

        return None



    def _generate_principal_disk_candidate_registry(self):
        undistorted_disk_centers = self._get_undistorted_disk_centers()

        ref_pt = np.array((0.5, 0.5))
        distances = np.linalg.norm(undistorted_disk_centers-ref_pt, axis=1)
        disk_idx = distances.argmin().item()
        
        ref_pt = undistorted_disk_centers[disk_idx]
        distances = np.linalg.norm(undistorted_disk_centers-ref_pt, axis=1)
        nn_distance = np.sort(distances)[1].item()
        nnn_distance = (np.sqrt(3)*nn_distance).item()

        principal_disk_candidate_registry = (distances <= 1.05*nnn_distance)

        return principal_disk_candidate_registry



    def _get_undistorted_disk_centers(self):
        undistorted_disks = self._cbed_pattern_generator._undistorted_disks

        undistorted_disk_centers = \
            tuple()
        for undistorted_disk in undistorted_disks:
            kwargs = \
                {"undistorted_disk": undistorted_disk}
            undistorted_disk_center = \
                self._get_undistorted_disk_center(**kwargs)
            undistorted_disk_centers += \
                (undistorted_disk_center,)
        undistorted_disk_centers = \
            np.array(undistorted_disk_centers)

        return undistorted_disk_centers



    def _get_undistorted_disk_center(self, undistorted_disk):
        undistorted_disk_core_attrs = \
            undistorted_disk.get_core_attrs(deep_copy=False)
        undistorted_disk_support = \
            undistorted_disk_core_attrs["support"]

        undistorted_disk_support_core_attrs = \
            undistorted_disk_support.get_core_attrs(deep_copy=False)
        undistorted_disk_center = \
            undistorted_disk_support_core_attrs["center"]

        return undistorted_disk_center



    def _initialize_and_cache_cbed_pattern_params(self):
        num_pixels_across_each_uncropped_pattern = \
            self._cbed_pattern_generator._sampling_grid_dims_in_pixels[0]
        ml_model_task_module = \
            self._cbed_pattern_generator._ml_model_task_module

        kwargs = {"path_to_ml_dataset": self._path_to_ml_training_dataset,
                  "entire_ml_dataset_is_to_be_cached": False,
                  "ml_data_values_are_to_be_checked": False,
                  "max_num_ml_data_instances_per_chunk": 32}
        ml_dataset = ml_model_task_module.MLDataset(**kwargs)

        kwargs = {"single_dim_slice": slice(0, 1), "device_name": "cpu"}
        ml_data_instances = ml_dataset.get_ml_data_instances(**kwargs)

        self._cropped_cbed_pattern_params = \
            {"cropping_window_dims_in_pixels": \
             2*(self._num_pixels_across_each_cropping_window,),
             "disk_boundary_sample_size": \
             ml_data_instances["principal_disk_boundary_pt_sets"].shape[1],
             "mask_frame": \
             4*(0,)}

        return None



    def generate(self):
        cropped_cbed_pattern_params = self._cropped_cbed_pattern_params

        generation_attempt_count = 0
        max_num_generation_attempts = 10
        cropped_cbed_pattern_generation_has_not_been_completed = True
        
        while cropped_cbed_pattern_generation_has_not_been_completed:
            try:
                param_name_subset = ("cbed_pattern",
                                     "principal_disk_idx",
                                     "cropping_window_center")
                for param_name in param_name_subset:
                    method_name = "_generate_{}".format(param_name)
                    method_alias = getattr(self, method_name)
                    cropped_cbed_pattern_params[param_name] = method_alias()
                    
                cls_alias = fakecbed.discretized.CroppedCBEDPattern
                kwargs = {**cropped_cbed_pattern_params,
                          "skip_validation_and_conversion": True}
                cropped_cbed_pattern = cls_alias(**kwargs)

                self._check_cropped_cbed_pattern(cropped_cbed_pattern)

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
                        _cropped_cbed_pattern_generator_err_msg_2
                    
                    args = ("", " ({})".format(max_num_generation_attempts))
                    err_msg = unformatted_err_msg.format(*args)                    
                    raise RuntimeError(err_msg)

        return cropped_cbed_pattern



    def _generate_cbed_pattern(self):
        cbed_pattern = self._cbed_pattern_generator.generate()

        return cbed_pattern



    def _generate_principal_disk_idx(self):
        cropped_cbed_pattern_params = self._cropped_cbed_pattern_params
        cbed_pattern = cropped_cbed_pattern_params["cbed_pattern"]

        disk_clipping_registry = \
            cbed_pattern.get_disk_clipping_registry(deep_copy=False)
        disk_clipping_registry = \
            disk_clipping_registry.numpy(force=True)
        
        principal_disk_candidate_registry = \
            self._principal_disk_candidate_registry

        disk_idx_subset = np.where((~disk_clipping_registry)
                                   * principal_disk_candidate_registry)[0]

        kwargs = {"a": disk_idx_subset}
        principal_disk_idx = self._rng.choice(**kwargs).item()

        return principal_disk_idx



    def _generate_cropping_window_center(self):
        cropped_cbed_pattern_params = self._cropped_cbed_pattern_params
        cbed_pattern = cropped_cbed_pattern_params["cbed_pattern"]
        principal_disk_idx = cropped_cbed_pattern_params["principal_disk_idx"]

        device = cbed_pattern.device

        kwargs = \
            {"cbed_pattern": cbed_pattern,
             "principal_disk_idx": principal_disk_idx}
        q_x_c, q_y_c = \
            self._generate_q_x_c_and_q_y_c_of_principal_disk(**kwargs)

        disk_supports = cbed_pattern.get_disk_supports(deep_copy=False)
        q_x, q_y = self._generate_q_x_and_q_y_of_cbed_pattern_signal(device)

        disk_support_COMs_shape = (cbed_pattern.num_disks, 2)
        disk_support_COMs = torch.zeros(disk_support_COMs_shape,
                                        device=device)

        disk_support_areas = disk_supports.sum(dim=(1, 2))

        disk_support_COMs[:, 0] = \
            (((q_x[None, :, :]*disk_supports).sum(dim=(1, 2))
              / (disk_support_areas + (disk_support_areas == 0)))
             + (disk_support_areas == 0)*1e6)
        disk_support_COMs[:, 1] = \
            (((q_y[None, :, :]*disk_supports).sum(dim=(1, 2))
              / (disk_support_areas + (disk_support_areas == 0)))
             + (disk_support_areas == 0)*1e6)

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



    def _generate_q_x_c_and_q_y_c_of_principal_disk(self,
                                                    cbed_pattern,
                                                    principal_disk_idx):
        cbed_pattern_core_attrs = cbed_pattern.get_core_attrs(deep_copy=False)
        
        undistorted_disks = cbed_pattern_core_attrs["undistorted_disks"]
        undistorted_disk = undistorted_disks[principal_disk_idx]

        distortion_model = cbed_pattern_core_attrs["distortion_model"]

        distortion_model_core_attrs = \
            distortion_model.get_core_attrs(deep_copy=False)
        coord_transform_params = \
            distortion_model_core_attrs["coord_transform_params"]
        
        undistorted_disk_core_attrs = \
            undistorted_disk.get_core_attrs(deep_copy=False)
        undistorted_disk_support = \
            undistorted_disk_core_attrs["support"]
        
        undistorted_disk_support_core_attrs = \
            undistorted_disk_support.get_core_attrs(deep_copy=False)
        u_x_c, u_y_c = \
            undistorted_disk_support_core_attrs["center"]

        device = cbed_pattern.device

        kwargs = {"u_x": torch.tensor(((u_x_c.item(),),), device=device),
                  "u_y": torch.tensor(((u_y_c.item(),),), device=device),
                  "coord_transform_params": coord_transform_params,
                  "device": device,
                  "skip_validation_and_conversion": True}
        q_x, q_y = distoptica.apply_coord_transform(**kwargs)

        q_x_c_and_q_y_c_of_principal_disk = (q_x[0, 0].item(), q_y[0, 0].item())

        return q_x_c_and_q_y_c_of_principal_disk



    def _generate_q_x_and_q_y_of_cbed_pattern_signal(self, device):
        num_pixels_across_each_cbed_pattern = \
            self._cbed_pattern_generator._sampling_grid_dims_in_pixels[0]

        size = num_pixels_across_each_cbed_pattern
        scale = 1/size
        offset = 0.5*scale

        pair_of_1d_coord_arrays = \
            (scale*torch.arange(size, device=device)+offset,
             1 - (scale*torch.arange(size, device=device)+offset))

        generate_q_x_and_q_y_of_cbed_pattern_signal = \
            torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")

        return generate_q_x_and_q_y_of_cbed_pattern_signal



    def _check_cropped_cbed_pattern(self, cropped_cbed_pattern):
        principal_disk_is_overlapping = \
            cropped_cbed_pattern.principal_disk_is_overlapping
        principal_disk_is_clipped = \
            cropped_cbed_pattern.principal_disk_is_clipped

        if principal_disk_is_overlapping or principal_disk_is_clipped:
            err_msg = _cropped_cbed_pattern_generator_err_msg_1
            raise ValueError(err_msg)

        return None



    def _generate_mask_frame(self, cropped_cbed_pattern):
        ml_model_task = self._cbed_pattern_generator._ml_model_task
        cropped_cbed_pattern_params = self._cropped_cbed_pattern_params

        num_pixels_across_each_cropping_window = \
            cropped_cbed_pattern_params["cropping_window_dims_in_pixels"][0]
        
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

        quadruple_1 = \
            np.array((max(bounding_box[0]-bounding_box_buffer[0], 0),
                      max(1-bounding_box[1]-bounding_box_buffer[1], 0),
                      max(bounding_box[2]-bounding_box_buffer[2], 0),
                      max(1-bounding_box[3]-bounding_box_buffer[3], 0)))

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



def _generate_argument_names():
    argument_names = ("ml_model_task",
                      "ml_input_image_width",
                      "disk_size_idx",
                      "disk_size",
                      "ml_dataset_idx",
                      "data_dir_1",
                      "data_dir_2")

    return argument_names



def parse_and_convert_cmd_line_args():
    accepted_ml_model_tasks = ("cbed/distortion/estimation",
                               "cbed/disk/localization",
                               "cbed/disk/segmentation")

    current_func_name = "parse_and_convert_cmd_line_args"

    try:
        parser = argparse.ArgumentParser()
        argument_names = _generate_argument_names()
        for argument_name in argument_names:
            parser.add_argument("--"+argument_name)
        args = parser.parse_args()
        ml_model_task = args.ml_model_task
        ml_input_image_width_in_pixels = int(args.ml_input_image_width)
        disk_size_idx = int(args.disk_size_idx)
        disk_size = args.disk_size
        ml_dataset_idx = int(args.ml_dataset_idx)
        path_to_data_dir_1 = args.data_dir_1
        path_to_data_dir_2 = args.data_dir_2

        if ((ml_model_task not in accepted_ml_model_tasks)
            or (disk_size_idx < 0)
            or (ml_dataset_idx < 0)):
            raise
    except:
        num_placeholders = len(accepted_ml_model_tasks)
        unformatted_partial_err_msg = (("``<{}>``, "*(num_placeholders-1))
                                       + "or ``<{}>``")
        args = accepted_ml_model_tasks
        partial_err_msg = unformatted_partial_err_msg.format(*args)
        
        unformatted_err_msg = globals()["_"+current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(partial_err_msg)
        raise SystemExit(err_msg)

    converted_cmd_line_args = \
        {"ml_model_task": ml_model_task,
         "ml_input_image_width_in_pixels": ml_input_image_width_in_pixels,
         "disk_size_idx": disk_size_idx,
         "disk_size": disk_size,
         "ml_dataset_idx": ml_dataset_idx,
         "path_to_data_dir_1": path_to_data_dir_1,
         "path_to_data_dir_2": path_to_data_dir_2}
    
    return converted_cmd_line_args



###########################
## Define error messages ##
###########################

_cbed_pattern_generator_err_msg_1 = \
    ("The CBED pattern must contain at least {} non-clipped CBED disks.")
_cbed_pattern_generator_err_msg_2 = \
    ("The CBED pattern generator{} has exceeded its programmed maximum number "
     "of attempts{} to generate a valid CBED pattern: see traceback for "
     "details.")

_cropped_cbed_pattern_generator_err_msg_1 = \
    ("The principal CBED disk of the cropped CBED pattern must not be clipped "
     "nor overlapping with any other CBED disks.")
_cropped_cbed_pattern_generator_err_msg_2 = \
    ("The cropped CBED pattern generator{} has exceeded its programmed maximum "
     "number of attempts{} to generate a valid cropped CBED pattern: see "
     "traceback for details.")

_parse_and_convert_cmd_line_args_err_msg_1 = \
    ("The correct form of the command is:\n"
     "\n"
     "    python execute_main_action_steps.py "
     "--ml_model_task=<ml_model_task> "
     "--ml_input_image_width=<ml_input_image_width> "
     "--disk_size_idx=<disk_size_idx> "
     "--disk_size=<disk_size> "
     "--ml_dataset_idx=<ml_dataset_idx> "
     "--data_dir_1=<data_dir_1> "
     "--data_dir_2=<data_dir_2>\n"
     "\n"
     "where ``<ml_model_task>`` must be {}; ``<ml_input_image_width>`` must be "
     "a nonnegative integer; ``<disk_size_idx>`` must be a nonnegative "
     "integer; ``<disk_size>`` must be ``small``, ``medium``, or ``large``; "
     "``<ml_dataset_idx>`` must be a nonnegative integer; ``<data_dir_1>`` "
     "must be a valid absolute path to a valid existing directory or one to be "
     "created; and ``<data_dir_2>`` must be the absolute path to a valid "
     "directory.")



#########################
## Main body of script ##
#########################

# Parse the command line arguments.
converted_cmd_line_args = \
    parse_and_convert_cmd_line_args()
ml_model_task = \
    converted_cmd_line_args["ml_model_task"]
ml_input_image_width_in_pixels = \
    converted_cmd_line_args["ml_input_image_width_in_pixels"]
disk_size_idx = \
    converted_cmd_line_args["disk_size_idx"]
disk_size = \
    converted_cmd_line_args["disk_size"]
ml_dataset_idx = \
    converted_cmd_line_args["ml_dataset_idx"]
path_to_data_dir_1 = \
    converted_cmd_line_args["path_to_data_dir_1"]
path_to_data_dir_2 = \
    converted_cmd_line_args["path_to_data_dir_2"]



# Select the ``emicroml`` submodule required to generate a ML dataset that is
# appropriate to the specified ML model task. Also, select the RNG seed
# according to the specified ML dataset index and disk size index.
module_name = "emicroml.modelling.{}".format(ml_model_task).replace("/", ".")
ml_model_task_module = sys.modules[module_name]

rng_seed = disk_size_idx + ml_dataset_idx + 100000



# Construct the "fake" CBED pattern generator.
kwargs = {"ml_model_task": \
          ml_model_task,
          "path_to_stem_multislice_sim_intensity_output": \
          path_to_data_dir_2 + "/stem_sim_intensity_output.h5",
          "max_num_disks_in_any_cbed_pattern": \
          90,
          "rng_seed": \
          rng_seed,
          "device_name": \
          None,
          "path_to_stem_multislice_sim_params": \
          path_to_data_dir_2 + "/stem_sim_params.json"}
if ml_model_task == "cbed/distortion/estimation":
    cls_name = "CBEDPatternGenerator"
else:
    ml_input_image_width = ml_input_image_width_in_pixels
    unformatted_path = (path_to_data_dir_1
                        + "/ml_datasets"
                        + "/ml_datasets_with"
                        + "_{}_pixel_wide_cropped_cbed_patterns"
                        + "/ml_dataset_for_training.h5")
    path_to_ml_training_dataset = unformatted_path.format(ml_input_image_width)

    kwargs = {**kwargs,
              "path_to_ml_training_dataset": \
              path_to_ml_training_dataset,
              "num_pixels_across_each_cropping_window": \
              ml_input_image_width_in_pixels}
    cls_name = "CroppedCBEDPatternGenerator"
cls_alias = globals()[cls_name]
pattern_generator = cls_alias(**kwargs)



# Generate and save the ML dataset.
unformatted_partial_path = ("/ml_datasets_with"
                            "_{}_pixel_wide_cropped_cbed_patterns")
partial_path = (unformatted_partial_path.format(ml_input_image_width_in_pixels)
                * (ml_model_task != "cbed/distortion/estimation"))

cbed_pattern_descriptor = "cropped_" * ("cbed/disk" in ml_model_task)
sample_name = "MoS2_on_amorphous_C"

unformatted_output_filename = (path_to_data_dir_1
                               + "/ml_datasets"
                               + "{}"
                               + "/ml_datasets_for_ml_model_test_set_1"
                               + "/ml_datasets_with_{}cbed_patterns_of_{}"
                               + "/ml_datasets_with_{}_sized_disks"
                               + "/ml_dataset_{}.h5")
output_filename = unformatted_output_filename.format(partial_path,
                                                     cbed_pattern_descriptor,
                                                     sample_name,
                                                     disk_size,
                                                     ml_dataset_idx)

num_patterns = 2880

kwargs = {"output_filename": output_filename,
          "max_num_ml_data_instances_per_file_update": 288}
if ml_model_task == "cbed/distortion/estimation":
    kwargs = {**kwargs,
              "num_cbed_patterns": \
              num_patterns,
              "cbed_pattern_generator": \
              pattern_generator,
              "max_num_disks_in_any_cbed_pattern": \
              pattern_generator.max_num_disks_in_any_cbed_pattern}
else:
    kwargs = {**kwargs,
              "num_cropped_cbed_patterns": \
              num_patterns,
              "cropped_cbed_pattern_generator": \
              pattern_generator,
              "resolution_level_of_disk_boundary_sample_size": \
              pattern_generator.resolution_level_of_disk_boundary_sample_size}
ml_model_task_module.generate_and_save_ml_dataset(**kwargs)
