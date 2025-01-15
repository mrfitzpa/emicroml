"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For creating path objects.
import pathlib

# For deserializing JSON objects.
import json



# For general array handling.
import numpy as np

# For running the STEM simulation.
import prismatique

# For postprocessing diffraction patterns.
import empix

# For modelling electron guns, lenses, and probes.
import embeam



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2023"
__credits__      = ["Matthew Fitzpatrick"]
__maintainer__   = "Matthew Fitzpatrick"
__email__        = "mrfitzpa@uvic.ca"
__status__       = "Development"



###############################################
## Define classes, functions, and contstants ##
###############################################



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

# Parse command line arguments.
parser = argparse.ArgumentParser()
argument_names = ("disk_size", "data_dir_1", "data_dir_2")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
disk_size = args.disk_size
path_to_data_dir_1 = args.data_dir_1
path_to_data_dir_2 = args.data_dir_2



# The worker parameters.
path_to_worker_params = (path_to_data_dir_1
                         + "/potential_slice_generator_output"
                         + "/worker_params.json")
worker_params = prismatique.worker.Params.load(path_to_worker_params)



# Specify the sample.
path_to_potential_slices = (path_to_data_dir_1
                            + "/potential_slice_generator_output"
                            + "/potential_slices_of_subset_0.h5")
path_to_sample_model_params = (path_to_data_dir_1
                               + "/potential_slice_generator_output"
                               + "/sample_model_params.json")

sample_model_params = \
    prismatique.sample.ModelParams.load(path_to_sample_model_params)
thermal_params = \
    sample_model_params.core_attrs["thermal_params"]
num_frozen_phonon_configs_per_subset = \
    thermal_params.core_attrs["num_frozen_phonon_configs_per_subset"]
discretization_params = \
    sample_model_params.core_attrs["discretization_params"]
interpolation_factors = \
    discretization_params.core_attrs["interpolation_factors"]

kwargs = {"filenames": \
          (path_to_potential_slices,),
          "interpolation_factors": \
          interpolation_factors,
          "max_num_frozen_phonon_configs_per_subset": \
          num_frozen_phonon_configs_per_subset}

sample_specification = prismatique.sample.PotentialSliceSubsetIDs(**kwargs)



# The simulation parameters related to the modelling of the electron gun.
kwargs = {"mean_beam_energy": 20,  # In keV.
          "intrinsic_energy_spread": 0.6e-3,  # In keV.
          "accel_voltage_spread": 0.0}  # In keV.
gun_model_params = embeam.gun.ModelParams(**kwargs)



# The coherent lens aberrations.
Delta_X, Delta_Y, _ = \
    prismatique.sample.supercell_dims(sample_specification)  # In Å.
N_x, N_y = \
    prismatique.sample.supercell_xy_dims_in_pixels(sample_specification)

target_cbed_disk_radii_in_fractional_coords = \
    {"small": 1/35, "medium": (1/35+1/10)/2, "large": 1/10}
target_cbed_disk_radius_in_fractional_coords = \
    target_cbed_disk_radii_in_fractional_coords[disk_size]

mean_beam_energy = gun_model_params.core_attrs["mean_beam_energy"]
wavelength = embeam.wavelength(mean_beam_energy)  # In Å.

convergence_semiangle = (target_cbed_disk_radius_in_fractional_coords
                         * (N_x/8) * (1/Delta_X)
                         * wavelength * 1e3)  # In mrads.

C_s = 8  # Spherical aberration strength in mm.
C_c = 8  # Chromatic aberration strength in mm.

filename = path_to_data_dir_2 + "/sample_model_params_subset.json"
with open(filename, 'r') as file_obj:
    serializable_rep = json.load(file_obj)
sample_model_params_subset = serializable_rep
a_MoS2 = sample_model_params_subset["a_MoS2"]  # "a" lattice parameter of MoS2.

nearest_neighbour_distance_between_atoms = a_MoS2/np.sqrt(3)  # In Å.
target_beam_radius = 3 * nearest_neighbour_distance_between_atoms  # In Å.

Delta_f_opt = -(1/2) * (C_s*1e7) * ((convergence_semiangle*1e-3)**2)  # In Å.
Delta_f = Delta_f_opt - target_beam_radius/(convergence_semiangle*1e-3)  # In Å.

C_2_0_mag = (2*np.pi*Delta_f)/(2*wavelength)  # In dimensionless units.
C_4_0_mag = (2*np.pi*(C_s*1e7))/(4*wavelength)  # In dimensionless units.

defocus_aberration = embeam.coherent.Aberration(m=2, 
                                                n=0, 
                                                C_mag=C_2_0_mag, 
                                                C_ang=0)
spherical_aberration = embeam.coherent.Aberration(m=4, 
                                                  n=0, 
                                                  C_mag=C_4_0_mag, 
                                                  C_ang=0)
coherent_aberrations = (defocus_aberration, spherical_aberration)



# The model used for both the probe forming and objective lenses.
kwargs = {"coherent_aberrations": coherent_aberrations,
          "chromatic_aberration_coef": C_c,
          "mean_current": 50,  # In pA.
          "std_dev_current": 0.0}  # In pA.
lens_model_params = embeam.lens.ModelParams(**kwargs)



# The probe model.
kwargs = {"lens_model_params": lens_model_params,
          "gun_model_params": gun_model_params,
          "convergence_semiangle": convergence_semiangle,
          "defocal_offset_supersampling": 9}
probe_model_params = embeam.stem.probe.ModelParams(**kwargs)



# The probe scan pattern.
num_probe_positions_across_scan_pattern = 1
x_step_size = 1.01 * (Delta_X/num_probe_positions_across_scan_pattern)  # In Å.
y_step_size = 1.01 * (Delta_Y/num_probe_positions_across_scan_pattern)  # In Å.
window = (0, 1, 0, 1)  # Dimensionless.
rng_seeds = {"small": 750, "medium": 751, "large": 752}

kwargs = {"step_size": (x_step_size, y_step_size),
          "window": window,
          "jitter": 0.1,  # Dimensionless.
          "rng_seed": rng_seeds[disk_size]}
scan_config = prismatique.scan.rectangular.Params(**kwargs)



# The STEM system model.
kwargs = {"sample_specification": sample_specification,
          "probe_model_params": probe_model_params,
          "specimen_tilt": (0, 0),  # In mrads.
          "scan_config": scan_config}
stem_system_model_params = prismatique.stem.system.ModelParams(**kwargs)



# The simulation parameters related to the generation of the convergent beam
# electron diffraction patterns.
new_signal_space_offsets = (-(1/Delta_X) * ((N_x//2)//2),
                            (1/Delta_X) * ((N_x//2-1)//2))
kwargs = {"new_signal_space_sizes": (N_x//2, N_x//2), 
          "new_signal_space_scales": (1/Delta_X, -1/Delta_X),
          "new_signal_space_offsets": new_signal_space_offsets,
          "spline_degrees": (1, 1), 
          "interpolate_polar_cmpnts": False}
optional_resampling_params = empix.OptionalResamplingParams(**kwargs)

kwargs = {"center": (0, 0),
          "window_dims": (N_x//4, N_x//4),
          "pad_mode": "zeros",
          "apply_symmetric_mask": True}
optional_cropping_params = empix.OptionalCroppingParams(**kwargs)

postprocessing_seq = (optional_resampling_params, optional_cropping_params)

kwargs = {"postprocessing_seq": postprocessing_seq,
          "avg_num_electrons_per_postprocessed_dp": 1,
          "apply_shot_noise": False,
          "save_wavefunctions": False,
          "save_final_intensity": True}
cbed_params = prismatique.cbed.Params(**kwargs)



# The simulation output parameters.
output_dirname = (path_to_data_dir_1
                  + "/cbed_pattern_generator_output"
                  + "/patterns_with_{}_sized_disks").format(disk_size)
kwargs = {"output_dirname": output_dirname,
          "max_data_size": 185e9,  # In bytes.
          "cbed_params": cbed_params,
          "radial_step_size_for_3d_stem": 0,  # In mrads.
          "radial_range_for_2d_stem": (0, 0),  # In mrads.
          "save_com": False,
          "save_potential_slices": False}
base_output_params = prismatique.stem.output.base.Params(**kwargs)

kwargs = {"num_slices_per_output": 1,
          "z_start_output": float("inf")}  # In Å
alg_specific_output_params = prismatique.stem.output.multislice.Params(**kwargs)

kwargs = {"base_params": base_output_params,
          "alg_specific_params": alg_specific_output_params}
output_params = prismatique.stem.output.Params(**kwargs)



# Group all the simulation parameters together.
kwargs = {"stem_system_model_params": stem_system_model_params,
          "output_params": output_params,
          "worker_params": worker_params}
sim_params = prismatique.stem.sim.Params(**kwargs)



# Run simulation.
prismatique.stem.sim.run(sim_params)
