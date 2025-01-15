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



###############################################
## Define classes, functions, and contstants ##
###############################################

def _parse_and_convert_cmd_line_args():
    accepted_actions = ("generate_atomic_coords",
                        "generate_potential_slices",
                        "generate_cbed_pattern_sets")

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--action", default=accepted_actions[0])
        parser.add_argument("--use_slurm", default="no")
        args = parser.parse_args()
        action = args.action
        use_slurm = args.use_slurm
        
        if (action not in accepted_actions) or (use_slurm not in ("yes", "no")):
            raise
    except:
        raise SystemExit(_parse_and_convert_cmd_line_args_err_msg_1)

    converted_cmd_line_args = {"action": action, "use_slurm": use_slurm}
    
    return converted_cmd_line_args



###########################
## Define error messages ##
###########################

_parse_and_convert_cmd_line_args_err_msg_1 = \
    ("The correct form of the command should be:\n"
     "\n"
     "    python execute_action.py "
     "--action=<action> --use_slurm=<use_slurm>\n"
     "\n"
     "where ``<action>`` can be "
     "``generate_atomic_coords``, "
     "``generate_potential_slices``, or "
     "``generate_cbed_pattern_sets``; and ``<use_slurm>`` can be either "
     "``yes`` or ``no``.")



#########################
## Main body of script ##
#########################

converted_cmd_line_args = _parse_and_convert_cmd_line_args()
action = converted_cmd_line_args["action"]
use_slurm = converted_cmd_line_args["use_slurm"]

path_to_current_script = pathlib.Path(os.path.realpath(__file__))
path_to_data_dir_1 = str(path_to_current_script.parents[1]) + "/data"
path_to_repo_root = str(path_to_current_script.parents[6])
script_to_execute = (str(path_to_current_script.parent)
                     + "/" + action + "/execute_all_action_steps.py")

unformatted_cmd_str = ("python {} "
                       "--data_dir_1={} "
                       "--repo_root={} "
                       "--use_slurm={}")

cmd_str = unformatted_cmd_str.format(script_to_execute,
                                     path_to_data_dir_1,
                                     path_to_repo_root,
                                     use_slurm)
os.system(cmd_str)
