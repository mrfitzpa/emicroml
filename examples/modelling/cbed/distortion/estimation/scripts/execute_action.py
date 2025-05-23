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
"""A script for training and testing machine learning (ML) models for distortion
estimation in convergent beam electron diffraction (CBED). The ML models that
can be used for distortion estimation are described in the documentation for the
class :class:`emicroml.modelling.cbed.distortion.estimation.MLModel`.

This script can be used to perform a variety of actions. The correct form of the
command to run the script is::

  python execute_action.py --action=<action> --use_slurm=<use_slurm>

where ``<action>`` is one of a set of accepted strings that specifies the action
to be performed, and ``<use_slurm>`` is either ``yes`` or ``no``. If
``<use_slurm>`` equals ``yes`` and the SLURM workload manager is available on
the server from which you intend to run the script, then the action will be
performed as a SLURM job. If ``<use_slurm>`` is equal to ``no``, then the action
will be performed locally without using a SLURM workload manager. ``<action>``
can be equal to ``generate_ml_datasets_for_training_and_validation``,
``combine_ml_datasets_for_training_and_validation_then_split``,
``train_ml_model_set``, ``generate_ml_datasets_for_ml_model_test_set_1``,
``combine_ml_datasets_for_ml_model_test_set_1``, or ``run_ml_model_test_set_1``.
We describe below in more detail each action that can be performed.

If ``<action>`` equals ``generate_ml_datasets_for_training_and_validation``,
then the script will generate 10 ML datasets that can be used to train and/or
evaluate ML models for distortion estimation in CBED. Let
``<top_level_data_dir>`` be
``<root>/examples/modelling/cbed/distortion/estimation/data``, where ``<root>``
be the path to the root of the git repository. For every nonnegative integer
``<k>`` less than 10, the ``<k>``th ML dataset is stored in the HDF5 file at the
file path
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_<k>.h5``.
The file structure of each HDF5 file storing an ML dataset is described in the
documentation for the function
:func:`emicroml.modelling.cbed.distortion.estimation.generate_and_save_ml_dataset`.
Each ML dataset contains 11520 ML data instances, where each ML data instance
stores a :math:`512 \times 512` "fake" CBED pattern containing at most 90 CBED
disks.

If ``<action>`` equals
``combine_ml_datasets_for_training_and_validation_then_split``, then the script
will take as input the 10 ML datasets generated from the previous action,
assuming the previous action had already been executed, combine said input ML
datasets, and then subsequently split the resulting ML dataset into two output
ML datasets: one intended for training ML models, the other for validating ML
models. Upon successful completion of the script, approximately 80 percent of
the input ML data instances are stored in the output ML dataset intended for
training ML models, and the remaining input ML data instances are stored in the
output ML dataset intented for validating ML models. Moreover, upon successful
completion, the files storing the input ML datasets are deleted. The output ML
datasets intended for training and validating the ML models are stored in the
HDF5 files at the file paths
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_training.h5``
and
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_validation.h5``
respectively.

If ``<action>`` equals ``train_ml_model_set``, then the script will train a
single ML model using the training and validation ML datasets generated from the
previous action, assuming the previous action had already been executed. Note
that the ML model set being alluded to here is a trivial set containing only one
ML model. Upon successful completion of the script, a dictionary representation
of the ML model after training is saved to a file at the file path
``<top_level_data_dir>/ml_models/ml_model_0/ml_model_at_lr_step_<last_lr_step>.pth``,
where ``<last_lr_step>`` is an integer indicating the last learning rate step in
the ML model training procedure. Additionally, the ML model training summary
output data file is saved to
``<top_level_data_dir>/ml_models/ml_model_0/ml_model_training_summary_output_data.h5"``.

If ``<action>`` equals ``generate_ml_datasets_for_ml_model_test_set_1``, then
the script will generate a set of 3 ML datasets that can be used to test ML
models for distortion estimation in CBED. This action depends on the output data
generated by running the script
``<root>/examples/modelling/cbed/simulation/MoS2_on_amorphous_C/scripts/execute_action.py``
with the command line argument ``--action`` of said script set to
``generate_cbed_pattern_sets``. Hence, one must execute the action
``generate_cbed_pattern_sets`` of the other script before executing the action
``generate_ml_datasets_for_ml_model_test_set_1`` of the current script. Note
that the action ``generate_cbed_pattern_sets`` depends on other actions as
well. See the summary documentation for the script
``<root>/examples/modelling/cbed/simulation/MoS2_on_amorphous_C/scripts/execute_action.py``
for more details on that matter. Upon successful completion of the action
``generate_ml_datasets_for_ml_model_test_set_1`` of the current script, for
every string ``<disk_size>`` in the sequence ``(small, medium, large)``, an ML
dataset is stored in the HDF5 file at the file path
``<top_level_data_dir>/ml_datasets/ml_datasets_for_ml_model_test_set_1/ml_datasets_with_cbed_patterns_of_MoS2_on_amorphous_C/ml_datasets_with_<disk_size>_sized_disks/ml_dataset_0.h5``,
where the ML dataset contains _ ML data instances, with each ML data instance
storing a :math:`512 \times 512` "fake" CBED pattern obtained by randomly
distorting the same undistorted CBED pattern with ``<disk_size>``-sized CBED
disks. For every string ``<disk_size>`` in the sequence ``(small, medium,
large)``, the undistorted CBED pattern with ``<disk_size>``-sized CBED disks is
the final CBED pattern of a CBED experiment simulated via the multislice
technique, where the sample is a 5-layer :math:`\text{MoS}_2` thin film with a
:math:`0.5 \ \text{nm}` thick layer of amorphous carbon (C). By small-, medium-,
and large-sized CBED disks, we mean CBED disks with radii equal roughly to 1/35,
(1/35+1/10)/2, and 1/10 in units of the image width, respectively. 

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

def parse_and_convert_cmd_line_args():
    accepted_actions = \
        ("generate_ml_datasets_for_training_and_validation",
         "combine_ml_datasets_for_training_and_validation_then_split",
         "train_ml_model_set",
         "generate_ml_datasets_for_ml_model_test_set_1",
         "combine_ml_datasets_for_ml_model_test_set_1",
         "run_ml_model_test_set_1")

    current_func_name = "parse_and_convert_cmd_line_args"

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
        num_placeholders = len(accepted_actions)
        unformatted_partial_err_msg = (("``{}``, "*(num_placeholders-1))
                                       + "or ``{}``")
        args = accepted_actions
        partial_err_msg = unformatted_partial_err_msg.format(*args)
        
        unformatted_err_msg = globals()["_"+current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(partial_err_msg)
        raise SystemExit(err_msg)

    converted_cmd_line_args = {"action": action, "use_slurm": use_slurm}
    
    return converted_cmd_line_args



###########################
## Define error messages ##
###########################

_parse_and_convert_cmd_line_args_err_msg_1 = \
    ("The correct form of the command is:\n"
     "\n"
     "    python execute_action.py "
     "--action=<action> --use_slurm=<use_slurm>\n"
     "\n"
     "where ``<action>`` can be {}; and ``<use_slurm>`` can be either ``yes`` "
     "or ``no``.")



#########################
## Main body of script ##
#########################

# Parse the command line arguments.
converted_cmd_line_args = parse_and_convert_cmd_line_args()
action = converted_cmd_line_args["action"]
use_slurm = converted_cmd_line_args["use_slurm"]



# Get the path to the script that executes the remainder of the action. This
# path is equal to
# ``<path_to_directory_containing_current_script>/../../../common/scripts/execute_action.py``,
# where ``<path_to_directory_containing_current_script>`` is the path to the
# directory containing directly the current script.
path_to_current_script = pathlib.Path(os.path.realpath(__file__))
path_to_calling_script_of_script_to_execute = str(path_to_current_script)
path_to_script_to_execute = (str(path_to_current_script.parents[3])
                             + "/common/scripts/execute_action.py")


# Execute the script at ``path_to_script_to_execute``.
unformatted_cmd_str = ("python {} "
                       "--calling_script={} "
                       "--action={} "
                       "--use_slurm={}")
args = (path_to_script_to_execute,
        path_to_calling_script_of_script_to_execute,
        action,
        use_slurm)
cmd_str = unformatted_cmd_str.format(*args)
os.system(cmd_str)
