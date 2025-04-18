"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For creating path objects.
import pathlib

# For getting the path to current script and for executing other scripts.
import os



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



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

parser = argparse.ArgumentParser()
argument_names = ("ml_model_task", "data_dir_1", "repo_root", "use_slurm")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
path_to_data_dir_1 = args.data_dir_1
path_to_repo_root = args.repo_root
use_slurm = args.use_slurm

path_to_current_script = pathlib.Path(os.path.realpath(__file__))
script_to_execute = (str(path_to_current_script.parents[1])
                     + "/generate_ml_dataset_for_ml_model_test_set_1"
                     + "/execute_all_action_steps.py")

unformatted_cmd_str = ("python {} "
                       "--ml_model_task={} "
                       "--disk_size_idx={} "
                       "--disk_size={} "
                       "--ml_dataset_idx={} "
                       "--data_dir_1={} "
                       "--repo_root={} "
                       "--use_slurm={}")

disk_sizes = ("small", "medium", "large")
num_ml_datasets = 1
for disk_size_idx, disk_size in enumerate(disk_sizes):
    for ml_dataset_idx in range(num_ml_datasets):
        cmd_str = unformatted_cmd_str.format(script_to_execute,
                                             ml_model_task,
                                             disk_size_idx,
                                             disk_size,
                                             ml_dataset_idx,
                                             path_to_data_dir_1,
                                             path_to_repo_root,
                                             use_slurm)
        os.system(cmd_str)
