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
estimation in CBED. The ML models that can be used for distortion estimation are
described in the documentation for the class
:class:`emicroml.modelling.cbed.distortion.estimation.MLModel`.

This script can be used to perform a variety of tasks. The correct form of the
command to run the script is::

  python execute_action.py --action=<action> --use_slurm=<use_slurm>

where ``<action>`` is one of a set of accepted strings that specifies the task
to be performed, and ``<use_slurm>`` is either ``yes`` or ``no``. If
``<use_slurm>`` equals ``yes`` and the SLURM workload manager is available on
the server from which you intend to run the script, then the task will be
performed as a SLURM job. If ``<use_slurm>`` is equal to ``no``, then the task
will be performed locally without using a SLURM workload manager. ``<action>``
can be equal to ``generate_ml_datasets_for_training_and_validation``,
``combine_ml_datasets_for_training_and_validation_then_split``,
``train_ml_model_set``, ``generate_ml_datasets_for_ml_model_test_set_1``,
``combine_ml_datasets_for_ml_model_test_set_1``, or ``run_ml_model_test_set_1``.
We describe below in more detail each task that can be performed.

If ``<action>`` equals ``generate_ml_datasets_for_training_and_validation``,
then the script will generate 10 ML datasets that can be used to train and/or
evaluate ML models for distortion estimation in CBED. Let
``<top_level_data_dir>`` be
``<root>/examples/modelling/cbed/distortion/estimation/data``, where ``<root>``
be the path to the root of the git repository. For every nonnegative integer <k>
less than 10, the <k>th ML dataset is stored in the HDF5 file at the file path
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_<k>.h5``.
The file structure of each HDF5 file storing an ML dataset is described in the
documentation for the function
:func:`emicroml.modelling.cbed.distortion.estimation.generate_and_save_ml_dataset`.
Each ML dataset contains 2880 ML data instances, where each ML data instance
stores a :math:`512 \times 512` "fake" CBED pattern containing at most 90 CBED
disks.

If ``<action>`` equals
``combine_ml_datasets_for_training_and_validation_then_split``, then the script
will take as input the 10 ML datasets from the previous action, assuming the
previous action had already been executed, combine said input ML datasets, and
then subsequently split the resulting ML dataset into two output ML datasets:
one intended for training ML models, the other for validating ML models. Upon
successful completion of the script, approximately 80 percent of the input ML
data instances are stored in the output ML dataset intended for training ML
models, and the remaining input ML data instances are stored in the output ML
dataset intented for validating ML models. Moreover, upon successful completion,
the files storing the input ML datasets are deleted. The output ML datasets
intended for training and validating the ML models are stored in the HDF5 files
at the file paths
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_training.h5``
and
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_validation.h5``
respectively.

If ``<action>`` equals ``train_ml_model_set``, then the script will 

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

def _parse_and_convert_cmd_line_args():
    accepted_actions = \
        ("generate_ml_datasets_for_training_and_validation",
         "combine_ml_datasets_for_training_and_validation_then_split",
         "run_learning_rate_range_test_set",
         "train_ml_model_set",
         "generate_ml_datasets_for_ml_model_test_set_1",
         "combine_ml_datasets_for_ml_model_test_set_1",
         "run_ml_model_test_set_1")

    current_func_name = "_parse_and_convert_cmd_line_args"

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
        
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
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

converted_cmd_line_args = _parse_and_convert_cmd_line_args()
action = converted_cmd_line_args["action"]
use_slurm = converted_cmd_line_args["use_slurm"]

path_to_current_script = pathlib.Path(os.path.realpath(__file__))
calling_script_of_script_to_execute = str(path_to_current_script)
script_to_execute = (str(path_to_current_script.parents[3])
                     + "/_common/scripts/execute_action.py")

unformatted_cmd_str = ("python {} "
                       "--calling_script={} "
                       "--action={} "
                       "--use_slurm={}")
cmd_str = unformatted_cmd_str.format(script_to_execute,
                                     calling_script_of_script_to_execute,
                                     action,
                                     use_slurm)
os.system(cmd_str)
