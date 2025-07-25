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
"""A script that is called by various other scripts used for combining 3 given
sets of input machine learning (ML) datasets into 3 larger ML datasets, intended
for testing ML models for a specified task.

The action of combining ML datasets as mentioned above is broken down
effectively into multiple steps, which can be summarized as follows:

1. Set the parameters and prepare the input data required to execute the "main"
steps of the action.

2. Execute the main steps.

3. If necessary, move non-temporary output data that are generated from the main
steps to their expected final destinations.

4. If necessary, delete/remove any remaining temporary files or directories.

The main steps are executed by running the script with the basename
``execute_main_action_steps.py``, located in the same directory as the current
script.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For pattern matching.
import re

# For parsing command line arguments.
import argparse

# For creating path objects.
import pathlib

# For getting the path to current script and for executing other scripts.
import os

# For checking if the ``sbatch`` shell command exists on the machine.
import shutil



##############################################
## Define classes, functions, and constants ##
##############################################

def parse_overriding_sbatch_options_file(path_to_repo_root):
    overriding_sbatch_options = tuple()

    path_to_overriding_sbatch_options_file = \
        path_to_repo_root + "/overriding_sbatch_options.sh"
    overriding_sbatch_options_file_exists = \
        pathlib.Path(path_to_overriding_sbatch_options_file).is_file()

    if overriding_sbatch_options_file_exists:
        with open(path_to_overriding_sbatch_options_file, "r") as file_obj:
            for line in file_obj:
                stripped_line = line.strip()
                pattern = r"#SBATCH --[a-z]+(-[a-z]+)*=.*"
                
                if re.fullmatch(pattern, stripped_line):
                    overriding_sbatch_option = stripped_line[8:]
                    overriding_sbatch_options += (overriding_sbatch_option,)

    return overriding_sbatch_options



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

# Parse the command line arguments.
parser = argparse.ArgumentParser()
argument_names = ("ml_model_task", "data_dir_1", "repo_root", "use_slurm")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
path_to_data_dir_1 = args.data_dir_1
path_to_repo_root = args.repo_root
use_slurm = args.use_slurm



# Execute the script that executes the remainder of the action and that accords
# to the command line argument ``use_slurm``.
path_to_current_script = pathlib.Path(os.path.realpath(__file__))

if use_slurm == "yes":
    overwrite_slurm_tmpdir = ("true"
                              if (shutil.which("sbatch") is None)
                              else "false")

    overriding_sbatch_options = \
        parse_overriding_sbatch_options_file(path_to_repo_root)
    path_to_dir_containing_current_script = \
        str(path_to_current_script.parent)
    path_to_script_to_execute = \
        str(path_to_current_script.parent) + "/prepare_and_submit_slurm_job.sh"

    args = (overriding_sbatch_options*(shutil.which("sbatch") is not None)
            + (path_to_script_to_execute,
               path_to_dir_containing_current_script,
               path_to_repo_root,
               path_to_data_dir_1,
               ml_model_task,
               overwrite_slurm_tmpdir))
    partial_cmd_str = "bash" if (shutil.which("sbatch") is None) else "sbatch"
    unformatted_cmd_str = partial_cmd_str+(" {}"*len(args))
    cmd_str = unformatted_cmd_str.format(*args)
else:
    path_to_script_to_execute = (str(path_to_current_script.parent)
                                 + "/execute_main_action_steps.py")
    unformatted_cmd_str = ("python {} "
                           "--ml_model_task={} "
                           "--data_dir_1={}")
    cmd_str = unformatted_cmd_str.format(path_to_script_to_execute,
                                         ml_model_task,
                                         path_to_data_dir_1)
os.system(cmd_str)
