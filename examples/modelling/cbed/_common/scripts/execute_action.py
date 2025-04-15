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
"""A script that is called by various other scripts used for performing tasks
related to CBED.

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
argument_names = ("calling_script", "action", "use_slurm")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
calling_script_of_current_script = args.calling_script
action = args.action
use_slurm = args.use_slurm

up_2_levels_from_calling_script_of_current_script = \
    pathlib.Path(calling_script_of_current_script).parents[1]
path_to_data_dir_1 = \
    (str(up_2_levels_from_calling_script_of_current_script) + "/data")

path_to_current_script = pathlib.Path(os.path.realpath(__file__))

up_4_levels_from_current_script = path_to_current_script.parents[3]

method_alias = up_2_levels_from_calling_script_of_current_script.relative_to
ml_model_task = method_alias(up_4_levels_from_current_script)

path_to_repo_root = str(path_to_current_script.parents[5])

script_to_execute = (str(path_to_current_script.parent)
                     + "/" + action + "/execute_all_action_steps.py")

unformatted_cmd_str = ("python {} "
                       "--ml_model_task={} "
                       "--data_dir_1={} "
                       "--repo_root={} "
                       "--use_slurm={}")
cmd_str = unformatted_cmd_str.format(script_to_execute,
                                     ml_model_task,
                                     path_to_data_dir_1,
                                     path_to_repo_root,
                                     use_slurm)
os.system(cmd_str)
