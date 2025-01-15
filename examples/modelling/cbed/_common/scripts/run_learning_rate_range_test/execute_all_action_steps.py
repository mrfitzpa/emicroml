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

# For checking if the ``sbatch`` shell command exists on the machine.
import shutil



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2024"
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

parser = argparse.ArgumentParser()
argument_names = ("ml_model_task",
                  "test_idx",
                  "data_dir_1",
                  "repo_root",
                  "use_slurm")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
test_idx = int(args.test_idx)
path_to_data_dir_1 = args.data_dir_1
path_to_repo_root = args.repo_root
use_slurm = args.use_slurm

path_to_current_script = pathlib.Path(os.path.realpath(__file__))

if use_slurm == "yes":
    overwrite_slurm_tmpdir = ("true"
                              if (shutil.which("sbatch") is None)
                              else "false")

    path_to_dir_containing_current_script = str(path_to_current_script.parent)
    script_to_execute = (str(path_to_current_script.parent)
                         + "/prepare_and_submit_slurm_job.sh")
    args = (script_to_execute,
            path_to_dir_containing_current_script,
            path_to_repo_root,
            path_to_data_dir_1,
            ml_model_task,
            test_idx,
            overwrite_slurm_tmpdir)
    partial_cmd_str = "bash" if (shutil.which("sbatch") is None) else "sbatch"
    unformatted_cmd_str = partial_cmd_str+(" {}"*len(args))
    cmd_str = unformatted_cmd_str.format(*args)
else:
    script_to_execute = (str(path_to_current_script.parent)
                         + "/execute_main_action_steps.py")
    unformatted_cmd_str = ("python {} "
                           "--ml_model_task={} "
                           "--test_idx={} "
                           "--data_dir_1={}")
    cmd_str = unformatted_cmd_str.format(script_to_execute,
                                         ml_model_task,
                                         test_idx,
                                         path_to_data_dir_1)
os.system(cmd_str)
