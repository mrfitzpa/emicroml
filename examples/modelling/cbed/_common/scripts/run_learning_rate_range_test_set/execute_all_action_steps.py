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
__copyright__    = "Copyright 2024"
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
                     + "/run_learning_rate_range_test"
                     + "/execute_all_action_steps.py")

unformatted_cmd_str = ("python {} "
                       "--ml_model_task={} "
                       "--test_idx={} "
                       "--data_dir_1={} "
                       "--repo_root={} "
                       "--use_slurm={}")

if ml_model_task == "cbed/fg/segmentation":
    num_tests_to_run = 1
elif ml_model_task == "cbed/disk/segmentation":
    num_tests_to_run = 1
elif ml_model_task == "cbed/distortion/estimation":
    num_tests_to_run = 16

for test_idx in range(num_tests_to_run):
    cmd_str = unformatted_cmd_str.format(script_to_execute,
                                         ml_model_task,
                                         test_idx,
                                         path_to_data_dir_1,
                                         path_to_repo_root,
                                         use_slurm)
    os.system(cmd_str)
    print("\n\n\n")
