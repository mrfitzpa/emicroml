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



# For special math functions.
import numpy as np



# For generating the potential slices.
import prismatique



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
argument_names = ("data_dir_1", "data_dir_2")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
path_to_data_dir_1 = args.data_dir_1
path_to_data_dir_2 = args.data_dir_2



# The CPU worker parameters.
kwargs = {"enable_workers": True,
          "num_worker_threads": 32,
          "batch_size": 1,
          "early_stop_count": 100}
cpu_params = prismatique.worker.cpu.Params(**kwargs)

# The GPU worker parameters. Note that if no GPUs are available then the
# following parameters are essentially ignored.
kwargs = {"num_gpus": 4,
          "batch_size": 1,
          "data_transfer_mode": "auto",
          "num_streams_per_gpu": 3}
gpu_params = prismatique.worker.gpu.Params(**kwargs)

# Put the CPU and GPU worker parameters together.
kwargs = {"cpu_params": cpu_params, "gpu_params": gpu_params}
worker_params = prismatique.worker.Params(**kwargs)



# The simulation parameters related to the discretization of real-space and
# Fourier/k-space.
filename = path_to_data_dir_2 + "/sample_model_params_subset.json"
with open(filename, 'r') as file_obj:
    serializable_rep = json.load(file_obj)
sample_model_params_subset = serializable_rep

sample_supercell_reduced_xy_dims_in_pixels = \
    sample_model_params_subset["sample_supercell_reduced_xy_dims_in_pixels"]
interpolation_factors = \
    sample_model_params_subset["interpolation_factors"]

filename = path_to_data_dir_1 + "/atomic_coords.xyz"
with open(filename, "r") as file_obj:
    throw_away_line = file_obj.readline()
    line = file_obj.readline()
    Delta_Z = float(line.split()[2])

target_slice_thickness = 1  # In Å.
num_slices = int(np.ceil(Delta_Z / target_slice_thickness))

kwargs = {"z_supersampling": 16,
          "sample_supercell_reduced_xy_dims_in_pixels": \
          sample_supercell_reduced_xy_dims_in_pixels,
          "interpolation_factors": interpolation_factors,
          "num_slices": num_slices}
discretization_params = prismatique.discretization.Params(**kwargs)



# The simulation parameters related to the thermal properties of the sample and
# its environment.
kwargs = {"enable_thermal_effects": True,
          "num_frozen_phonon_configs_per_subset": 16,
          "num_subsets": 1,
          "rng_seed": 500}
thermal_params = prismatique.thermal.Params(**kwargs)



# The simulation parameters related to the modelling of the sample.
atomic_potential_extent = sample_model_params_subset["atomic_potential_extent"]

kwargs = {"atomic_coords_filename": path_to_data_dir_1 + "/atomic_coords.xyz",
          "unit_cell_tiling": (1, 1, 1),
          "discretization_params": discretization_params,
          "atomic_potential_extent": atomic_potential_extent,
          "thermal_params": thermal_params}
sample_model_params = prismatique.sample.ModelParams(**kwargs)



# Use the parameters above to generate the potential slices.
output_dirname = path_to_data_dir_1 + "/potential_slice_generator_output"
kwargs = {"sample_model_params": sample_model_params,
          "output_dirname": output_dirname,
          "max_data_size": 185*10**9,  # In bytes.
          "worker_params": worker_params}
prismatique.sample.generate_potential_slices(**kwargs)
