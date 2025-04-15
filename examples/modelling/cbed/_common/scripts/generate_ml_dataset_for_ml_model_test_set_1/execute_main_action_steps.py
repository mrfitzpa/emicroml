"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse



# For general array handling.
import numpy as np
import torch

# For calculating electron beam wavelengths given mean beam energies.
import embeam

# For generating fake CBED disks.
import fakecbed

# For loading HDF5 datasubsets.
import h5pywrappers

# For loading STEM simulation parameters.
import prismatique



# For generating CBED pattern datsets.
import emnn.modelling.cbed.distortion.estimation



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2023"
__credits__      = ["Matthew Fitzpatrick"]
__maintainer__   = "Matthew Fitzpatrick"
__email__        = "mrfitzpa@uvic.ca"
__status__       = "Development"



##############################################
## Define classes, functions, and constants ##
##############################################

class CBEDPatternGenerator():
    def __init__(self,
                 ml_model_task,
                 path_to_stem_multislice_sim_params,
                 path_to_stem_multislice_sim_intensity_output,
                 num_samples_across_each_pixel,
                 max_num_disks_in_any_cbed_pattern,
                 device_name,
                 rng_seed):
        self.ml_model_task = \
            ml_model_task
        self.path_to_stem_multislice_sim_intensity_output = \
            path_to_stem_multislice_sim_intensity_output
        self.num_samples_across_each_pixel = \
            num_samples_across_each_pixel
        self.max_num_disks_in_any_cbed_pattern = \
            max_num_disks_in_any_cbed_pattern
        self.device_name = \
            device_name
        self.rng_seed = \
            rng_seed

        kwargs = {"path_to_stem_multislice_sim_params": \
                  path_to_stem_multislice_sim_params}
        self.store_relevant_stem_multislice_sim_params(**kwargs)
        
        self.wavelength = embeam.wavelength(self.mean_beam_energy)
        
        self.store_stem_multislice_sim_intensity_pattern_signal()
        self.store_relevant_properties_of_stem_multislice_sim_intensity_output()

        self.base_cbed_pattern_generator = \
            self.generate_base_cbed_pattern_generator()
        self.undistorted_tds_model = \
            self.generate_undistorted_tds_model()

        self.initialize_and_cache_cbed_pattern_params()

        return None



    def store_relevant_stem_multislice_sim_params(
            self, path_to_stem_multislice_sim_params):
        stem_multislice_sim_params = \
            prismatique.stem.sim.Params.load(path_to_stem_multislice_sim_params)
        stem_system_model_params = \
            stem_multislice_sim_params.core_attrs["stem_system_model_params"]
        probe_model_params = \
            stem_system_model_params.core_attrs["probe_model_params"]
        gun_model_params = \
            probe_model_params.core_attrs["gun_model_params"]

        self.convergence_semiangle = \
            probe_model_params.core_attrs["convergence_semiangle"]  # In mrads.
        self.mean_beam_energy = \
            gun_model_params.core_attrs["mean_beam_energy"]  # in keVs.

        return None



    def store_stem_multislice_sim_intensity_pattern_signal(self):
        kwargs = {"filename": self.path_to_stem_multislice_sim_intensity_output,
                  "multi_dim_slice": (0, 0)}
        signal, _ = prismatique.load.cbed_intensity_patterns(**kwargs)
        self.stem_multislice_sim_intensity_pattern_signal = signal

        return None



    def store_relevant_properties_of_stem_multislice_sim_intensity_output(self):
        signal = self.stem_multislice_sim_intensity_pattern_signal

        num_pixels_across_cbed_pattern = signal.axes_manager.signal_shape[0]//2
        self.num_pixels_across_cbed_pattern = num_pixels_across_cbed_pattern

        k_x_axis_label = "$k_x$"
        self.k_x_offset = signal.axes_manager[k_x_axis_label].offset
        self.k_x_scale = signal.axes_manager[k_x_axis_label].scale
        self.k_x_size = signal.axes_manager[k_x_axis_label].size

        k_y_axis_label = "$k_y$"
        self.k_y_offset = signal.axes_manager[k_y_axis_label].offset
        self.k_y_scale = signal.axes_manager[k_y_axis_label].scale
        self.k_y_size = signal.axes_manager[k_y_axis_label].size

        return None



    def generate_base_cbed_pattern_generator(self):
        if ml_model_task == "cbed/distortion/estimation":
            module_alias = emnn.modelling.cbed.distortion.estimation
            cls_alias = module_alias.DefaultCBEDPatternGenerator
            kwargs = {"num_pixels_across_cbed_pattern": \
                      self.num_pixels_across_cbed_pattern,
                      "num_samples_across_each_pixel": \
                      self.num_samples_across_each_pixel,
                      "rng_seed": \
                      self.rng_seed,
                      "max_num_disks_in_any_cbed_pattern": \
                      self.max_num_disks_in_any_cbed_pattern,
                      "device_name": \
                      self.device_name}

        base_cbed_pattern_generator = cls_alias(**kwargs)

        return base_cbed_pattern_generator



    def generate_undistorted_tds_model(self):
        kwargs = {"center": (0.5, 0.5),
                  "width": 1,
                  "val_at_center": 0,
                  "functional_form": "gaussian"}
        tds_peak = fakecbed.shapes.Peak(**kwargs)

        kwargs = {"peak": tds_peak, "constant_background": 0}
        undistorted_tds_model = fakecbed.tds.Model(**kwargs)

        return undistorted_tds_model



    def initialize_and_cache_cbed_pattern_params(self):
        num_pixels_across_cbed_pattern = self.num_pixels_across_cbed_pattern
        num_samples_across_each_pixel = self.num_samples_across_each_pixel

        obj_alias = self.base_cbed_pattern_generator
        sigma_1, sigma_2 = obj_alias._generate_std_devs_of_gaussian_filters()

        kwargs = \
            {"undistorted_tds_model": self.undistorted_tds_model,
             "undistorted_disks": self.generate_undistorted_disks(),
             "undistorted_background_bands": tuple(),
             "disk_support_gaussian_filter_std_dev": sigma_1,
             "intra_disk_gaussian_filter_std_dev": sigma_2,
             "distortion_model": None,
             "apply_shot_noise": True,
             "cold_pixels": tuple(),
             "num_pixels_across_pattern": num_pixels_across_cbed_pattern,
             "num_samples_across_each_pixel": num_samples_across_each_pixel,
             "mask_frame": (0, 0, 0, 0)}
        self.cbed_pattern_params = \
            fakecbed.discretized.CBEDPatternParams(**kwargs)

        return None



    def generate_undistorted_disks(self):
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
        k_x_tiling_indices = self.calc_k_x_tiling_indices(q_1)
        k_y_tiling_indices = self.calc_k_y_tiling_indices(q_2)

        undistorted_disks = tuple()

        for k_x_tiling_idx in k_x_tiling_indices:
            for k_y_tiling_idx in k_y_tiling_indices:
                shift = k_x_tiling_idx*q_1 + k_y_tiling_idx*q_2
                current_disk_cell = np.array(tuple(delta_disk+shift
                                                   for delta_disk
                                                   in disk_unit_cell))

                for (k_x_c_support, k_y_c_support) in current_disk_cell:
                    kwargs = {"k_x_coord": k_x_c_support}
                    x_c_support = self.k_x_coord_to_x_coord(**kwargs)

                    kwargs = {"k_y_coord": k_y_c_support}
                    y_c_support = self.k_y_coord_to_y_coord(**kwargs)

                    kwargs = {"center": (x_c_support, y_c_support),
                              "radius": self.calc_R_support(),
                              "intra_disk_val": 1}
                    support = fakecbed.shapes.UniformDisk(**kwargs)

                    kwargs = {"support": support,
                              "intra_disk_shapes": tuple()}
                    NonUniformDisk = fakecbed.shapes.NonUniformDisk
                    undistorted_disk = NonUniformDisk(**kwargs)
                    undistorted_disks += (undistorted_disk,)

        return undistorted_disks



    def calc_k_x_tiling_indices(self, q_1):
        k_R_support = self.calc_k_R_support()

        q_1_norm = np.linalg.norm(q_1)

        k_x_offset = self.k_x_offset
        k_x_scale = self.k_x_scale
        k_x_size = self.k_x_size

        k_x_max_candidate_1 = (abs(k_x_offset)
                               + k_R_support)
        k_x_max_candidate_2 = (abs(k_x_offset + k_x_scale*(k_x_size-1))
                               + k_R_support)
        
        k_x_max = max(k_x_max_candidate_1, k_x_max_candidate_2)

        max_k_x_tiling_idx = int(k_x_max // q_1_norm)
        min_k_x_tiling_idx = -max_k_x_tiling_idx

        k_x_tiling_indices = range(min_k_x_tiling_idx, max_k_x_tiling_idx+1)

        return k_x_tiling_indices



    def calc_k_R_support(self):
        k_R_support = (self.convergence_semiangle/1000) / self.wavelength

        return k_R_support



    def calc_k_y_tiling_indices(self, q_2):
        k_R_support = self.calc_k_R_support()

        q_2_norm = np.linalg.norm(q_2)

        k_y_offset = self.k_y_offset
        k_y_scale = self.k_y_scale
        k_y_size = self.k_y_size

        k_y_max_candidate_1 = (abs(k_y_offset)
                               + k_R_support)
        k_y_max_candidate_2 = (abs(k_y_offset + k_y_scale*(k_y_size-1))
                               + k_R_support)
        
        k_y_max = max(k_y_max_candidate_1, k_y_max_candidate_2)

        max_k_y_tiling_idx = int(k_y_max // q_2_norm)
        min_k_y_tiling_idx = -max_k_y_tiling_idx

        k_y_tiling_indices = range(min_k_y_tiling_idx, max_k_y_tiling_idx+1)

        return k_y_tiling_indices



    def k_x_coord_to_x_coord(self, k_x_coord):
        x_offset = 0.5/(self.k_x_size/2)
        x_scale = 1/(self.k_x_size/2)

        n_x = self.k_x_coord_to_pixel_coord(k_x_coord)

        x_coord = x_offset + n_x*x_scale

        return x_coord



    def k_x_coord_to_pixel_coord(self, k_x_coord):
        pixel_coord = (k_x_coord-(self.k_x_offset/2))/self.k_x_scale
        
        return pixel_coord



    def k_y_coord_to_y_coord(self, k_y_coord):
        y_offset = 1-(1-0.5)/(self.k_y_size/2)
        y_scale = -1/(self.k_y_size/2)

        n_y = self.k_y_coord_to_pixel_coord(k_y_coord)

        y_coord = y_offset + n_y*y_scale

        return y_coord



    def k_y_coord_to_pixel_coord(self, k_y_coord):
        pixel_coord = (k_y_coord-(self.k_y_offset/2))/self.k_y_scale

        return pixel_coord



    def calc_R_support(self):
        k_R_support = self.calc_k_R_support()

        x_scale = 1/(self.k_x_size/2)
        R_support = (k_R_support/self.k_x_scale)*x_scale

        return R_support



    def generate(self):
        key_1 = \
            "distortion_model"
        distortion_model = \
            self.generate_distortion_model()
        self.cbed_pattern_params._checkable_obj._core_attrs[key_1] = \
            distortion_model

        key_2 = \
            "mask_frame"
        mask_frame = \
            self.generate_mask_frame(distortion_model)
        self.cbed_pattern_params._checkable_obj._core_attrs[key_2] = \
            mask_frame

        kwargs = \
            {"distortion_model": distortion_model}
        maskless_intensity_image_data = \
            self.generate_overriding_maskless_intensity_image_data(**kwargs)

        kwargs = {"cbed_pattern_params": self.cbed_pattern_params}
        cbed_pattern = self.base_cbed_pattern_generator.generate(**kwargs)

        kwargs = {"maskless_intensity_image_data": \
                  maskless_intensity_image_data}
        cbed_pattern.override_maskless_intensity_image_data(**kwargs)

        return cbed_pattern

    

    def generate_distortion_model(self):
        distortion_model = None

        while distortion_model is None:
            try:
                obj_alias = \
                    self.base_cbed_pattern_generator
                kwargs = \
                    {"undistorted_tds_model": self.undistorted_tds_model}
                distortion_model = \
                    obj_alias._generate_distortion_model(**kwargs)
            except:
                distortion_model = \
                    None

        return distortion_model



    def generate_mask_frame(self, distortion_model):
        obj_alias = self.base_cbed_pattern_generator

        kwargs = {"distortion_model": distortion_model}
        mask_frame = obj_alias._generate_mask_frame(**kwargs)

        return mask_frame



    def generate_overriding_maskless_intensity_image_data(self,
                                                          distortion_model):
        device = distortion_model._device
        signal = self.stem_multislice_sim_intensity_pattern_signal

        input_tensor_to_sample = torch.from_numpy(signal.data)
        input_tensor_to_sample = input_tensor_to_sample.to(device)
        input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)
        input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)

        hd_x_data = distortion_model._cached_u_x[0, 0]
        hd_y_data = distortion_model._cached_u_y[0, 0]

        grid_shape = (1,) + hd_x_data.shape + (2,)
        grid = torch.zeros(grid_shape,
                           dtype=hd_x_data.dtype,
                           device=hd_x_data.device)
        grid[0, :, :, 0] = hd_x_data-0.5
        grid[0, :, :, 1] = -(hd_y_data-0.5)

        kwargs = \
            {"input": input_tensor_to_sample,
             "grid": grid,
             "mode": "bilinear",
             "padding_mode": "zeros",
             "align_corners": False}
        maskless_intensity_image_data = \
            torch.nn.functional.grid_sample(**kwargs)[0, 0]
        maskless_intensity_image_data = \
            maskless_intensity_image_data.cpu().detach().numpy()

        return maskless_intensity_image_data



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

parser = argparse.ArgumentParser()
argument_names = ("ml_model_task",
                  "disk_size_idx",
                  "disk_size",
                  "ml_dataset_idx",
                  "data_dir_1",
                  "data_dir_2")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
disk_size_idx = int(args.disk_size_idx)
disk_size = args.disk_size
ml_dataset_idx = int(args.ml_dataset_idx)
path_to_data_dir_1 = args.data_dir_1
path_to_data_dir_2 = args.data_dir_2

if ml_model_task == "cbed/distortion/estimation":
    module_alias = emnn.modelling.cbed.distortion.estimation
    rng_seed = disk_size_idx + ml_dataset_idx + 100000

path_to_stem_multislice_sim_params = \
    path_to_data_dir_2 + "/stem_sim_params.json"
path_to_stem_multislice_sim_intensity_output = \
    path_to_data_dir_2 + "/stem_sim_intensity_output.h5"

kwargs = {"ml_model_task": \
          ml_model_task,
          "path_to_stem_multislice_sim_params": \
          path_to_stem_multislice_sim_params,
          "path_to_stem_multislice_sim_intensity_output": \
          path_to_stem_multislice_sim_intensity_output,
          "num_samples_across_each_pixel": \
          2**3,
          "max_num_disks_in_any_cbed_pattern": \
          300,
          "device_name": \
          None,
          "rng_seed": \
          rng_seed}
cbed_pattern_generator = CBEDPatternGenerator(**kwargs)

sample_name = "MoS2_on_amorphous_C"

unformatted_output_filename = (path_to_data_dir_1
                               + "/ml_datasets"
                               + "/ml_datasets_for_ml_model_test_set_1"
                               + "/ml_datasets_with_cbed_patterns_of_{}"
                               + "/ml_datasets_with_{}_sized_disks"
                               + "/ml_dataset_{}.h5")
output_filename = unformatted_output_filename.format(sample_name,
                                                     disk_size,
                                                     ml_dataset_idx)

kwargs = {"num_cbed_patterns":  2880,
          "cbed_pattern_generator": cbed_pattern_generator,
          "output_filename": output_filename,
          "max_num_ml_data_instances_per_file_update": 240}
if ml_model_task == "cbed/distortion/estimation":
    kwargs["max_num_disks_in_any_cbed_pattern"] = 90
    
module_alias.generate_and_save_ml_dataset(**kwargs)
