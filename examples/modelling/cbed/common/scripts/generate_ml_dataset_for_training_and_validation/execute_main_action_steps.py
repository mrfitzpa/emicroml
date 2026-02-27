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
r"""A script that is called by various other scripts used for generating
individual machine learning (ML) datasets that can be used to train and/or
evaluate ML models for a specified task.

The correct form of the command to run the script is::

  python execute_main_action_steps.py \
         --ml_model_task=<ml_model_task> \
         --ml_dataset_idx=<ml_dataset_idx> \
         --data_dir_1=<data_dir_1>

where ``<ml_model_task>`` is one of a set of accepted strings that specifies the
ML model task; ``<ml_dataset_idx>`` is an integer that is used to label the
individual ML dataset to be generated; and ``<data_dir_1>`` is the absolute path
to an existing directory or one to be created, within which the output data is
to be saved.

At the moment, the only accepted value of ``<ml_model_task>`` is
``cbed/distortion/estimation``, which specifies that the ML model task is
distortion estimation in CBED.

``<ml_dataset_idx>`` can be any nonnegative integer, and ``<data_dir_1>`` can be
any valid absolute path to any valid existing directory or one to be created.

The only non-temporary output data generated from this script is a single HDF5
file, which stores the ML dataset. Upon successful execution of the script, the
HDF5 file is saved to
``<data_dir_1>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_<ml_dataset_idx>.h5``.

This script uses the module
:mod:`emicroml.modelling.cbed.distortion.estimation`. It is recommended that you
consult the documentation of said module as you explore the remainder of this
script.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For accessing imported modules via their names stored as strings.
import sys



# For generating images and targets in ML datasets.
import emicroml.modelling.cbed.distortion.estimation
import emicroml.modelling.cbed.disk.localization
import emicroml.modelling.cbed.disk.segmentation



##############################################
## Define classes, functions, and constants ##
##############################################

def parse_and_convert_cmd_line_args():
    accepted_ml_model_tasks = ("cbed/distortion/estimation",
                               "cbed/disk/localization",
                               "cbed/disk/segmentation")

    current_func_name = "parse_and_convert_cmd_line_args"

    try:
        parser = argparse.ArgumentParser()
        argument_names = ("ml_model_task", "ml_dataset_idx", "data_dir_1")
        for argument_name in argument_names:
            parser.add_argument("--"+argument_name)
        args = parser.parse_args()
        ml_model_task = args.ml_model_task
        ml_dataset_idx = int(args.ml_dataset_idx)
        path_to_data_dir_1 = args.data_dir_1

        if ((ml_model_task not in accepted_ml_model_tasks)
            or (ml_dataset_idx < 0)):
            raise
    except:
        num_placeholders = len(accepted_ml_model_tasks)
        unformatted_partial_err_msg = (("``<{}>``, "*(num_placeholders-1))
                                       + "or ``<{}>``")
        args = accepted_ml_model_tasks
        partial_err_msg = unformatted_partial_err_msg.format(*args)
        
        unformatted_err_msg = globals()["_"+current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(partial_err_msg)
        raise SystemExit(err_msg)

    converted_cmd_line_args = {"ml_model_task": ml_model_task,
                               "ml_dataset_idx": ml_dataset_idx,
                               "path_to_data_dir_1": path_to_data_dir_1}
    
    return converted_cmd_line_args



###########################
## Define error messages ##
###########################

_parse_and_convert_cmd_line_args_err_msg_1 = \
    ("The correct form of the command is:\n"
     "\n"
     "    python execute_main_action_steps.py "
     "--ml_model_task=<ml_model_task> "
     "--ml_dataset_idx=<ml_dataset_idx> "
     "--data_dir_1=<data_dir_1>\n"
     "\n"
     "where ``<ml_model_task>`` must be {}; ``<ml_dataset_idx>`` must be a "
     "nonnegative integer; and ``<data_dir_1>`` must be a valid absolute path "
     "to a valid existing directory or one to be created.")



#########################
## Main body of script ##
#########################

# Parse the command line arguments.
converted_cmd_line_args = parse_and_convert_cmd_line_args()
ml_model_task = converted_cmd_line_args["ml_model_task"]
ml_dataset_idx = converted_cmd_line_args["ml_dataset_idx"]
path_to_data_dir_1 = converted_cmd_line_args["path_to_data_dir_1"]



# Select the ``emicroml`` submodule required to generate a ML dataset that is
# appropriate to the specified ML model task. 
module_name = "emicroml.modelling.{}".format(ml_model_task).replace("/", ".")
ml_model_task_module = sys.modules[module_name]

    

# Construct the "fake" CBED pattern generator.
num_pixels_across_each_cbed_pattern = 512
sampling_grid_dims_in_pixels = 2*(num_pixels_across_each_cbed_pattern,)

kwargs = {"num_pixels_across_each_cbed_pattern": \
          num_pixels_across_each_cbed_pattern,
          "rng_seed": \
          ml_dataset_idx + 4000,
          "sampling_grid_dims_in_pixels": \
          sampling_grid_dims_in_pixels,
          "least_squares_alg_params": \
          None,
          "device_name": \
          None}
if ml_model_task == "cbed/distortion/estimation":
    kwargs = {**kwargs,
              "max_num_disks_in_any_cbed_pattern": \
              90}
    cls_name = "DefaultCBEDPatternGenerator"
else:
    kwargs = {**kwargs,
              "max_num_disks_in_any_cbed_pattern": \
              10,
              "num_pixels_across_each_cropping_window": \
              num_pixels_across_each_cbed_pattern//4}
    cls_name = "DefaultCroppedCBEDPatternGenerator"
cls_alias = getattr(ml_model_task_module, cls_name)
pattern_generator = cls_alias(**kwargs)



# Generate and save the ML dataset.
unformatted_output_filename = (path_to_data_dir_1
                               + "/ml_datasets"
                               + "/ml_datasets_for_training_and_validation"
                               + "/ml_dataset_{}.h5")
output_filename = unformatted_output_filename.format(ml_dataset_idx)

# num_patterns = 11520
num_patterns = 2

kwargs = {"output_filename": output_filename,
          "max_num_ml_data_instances_per_file_update": 576}
if ml_model_task == "cbed/distortion/estimation":
    kwargs = {**kwargs,
              "num_cbed_patterns": \
              num_patterns,
              "cbed_pattern_generator": \
              pattern_generator,
              "max_num_disks_in_any_cbed_pattern": \
              pattern_generator.core_attrs["max_num_disks_in_any_cbed_pattern"]}
else:
    kwargs = {**kwargs,
              "num_cropped_cbed_patterns": \
              num_patterns,
              "cropped_cbed_pattern_generator": \
              pattern_generator}
ml_model_task_module.generate_and_save_ml_dataset(**kwargs)
