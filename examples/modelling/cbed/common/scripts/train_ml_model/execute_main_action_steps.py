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
r"""A script that is called by various other scripts used for training 
individual machine learning (ML) models for a specified task.

The correct form of the command to run the script is::

  python execute_main_action_steps.py \
         --ml_model_task=<ml_model_task> \
         --ml_model_idx=<ml_model_idx> \
         --data_dir_1=<data_dir_1>

where ``<ml_model_task>`` is one of a set of accepted strings that specifies the
ML model task; ``<ml_model_idx>`` is an integer that is used to label the
individual ML model to be trained; and ``<data_dir_1>`` is the absolute path to
the top-level data directory containing the input data for this script.

At the moment, the only accepted value of ``<ml_model_task>`` is
``cbed/distortion/estimation``, which specifies that the ML model task is
distortion estimation in CBED.

``<ml_model_idx>`` can be any nonnegative integer. 

``<data_dir_1>`` must be the absolute path to an existing directory that
contains the subdirectory ``ml_datasets``, which must contain directly the HDF5
files storing the input ML datasets. The basenames of the HDF5 files storing the
training and validation ML datasets are ``ml_dataset_for_training.h5`` and
``ml_dataset_for_validation.h5`` respectively.

Upon successful completion of the current script, a dictionary representation of
the ML model after training is saved to a file at the file path
``<data_dir_1>/ml_models/ml_model_<ml_model_idx>/ml_model_at_lr_step_<last_lr_step>.pth``,
where ``<last_lr_step>`` is an integer indicating the last learning rate step in
the ML model training procedure. Additionally, the ML model training summary
output data file is saved to
``<data_dir_1>/ml_models/ml_model_<ml_model_idx>/ml_model_training_summary_output_data.h5"``.
These are the only two non-temporary output data files generated from this
script.

This script uses the modules :mod:`emicroml.modelling.optimizers`,
:mod:`emicroml.modelling.lr.schedulers`, and
:mod:`emicroml.modelling.cbed.distortion.estimation`. It is recommended that you
consult the documentation of these modules as you explore the remainder of this
script.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For accessing imported modules via their names stored as strings.
import sys

# For setting Python's seed.
import random

# For creating path objects.
import pathlib

# For listing subdirectories.
import os

# For pattern matching.
import re



# For avoiding errors related to the ``mkl-service`` package. Note that
# ``numpy`` needs to be imported before ``torch``.
import numpy as np

# For setting the seed to the random-number-generator used in ``pytorch``.
import torch

# For loading objects from and saving objects to HDF5 files.
import h5pywrappers



# For training ML models.
import emicroml.modelling.optimizers
import emicroml.modelling.lr.schedulers
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
        argument_names = ("ml_model_task",
                          "ml_model_idx",
                          "data_dir_1",
                          "ml_training_dataset")
        for argument_name in argument_names:
            parser.add_argument("--"+argument_name)
        args = parser.parse_args()
        ml_model_task = args.ml_model_task
        ml_model_idx = int(args.ml_model_idx)
        path_to_data_dir_1 = args.data_dir_1
        path_to_ml_training_dataset = args.ml_training_dataset

        if ((ml_model_task not in accepted_ml_model_tasks)
            or (ml_model_idx < 0)):
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

    converted_cmd_line_args = \
        {"ml_model_task": ml_model_task,
         "ml_model_idx": ml_model_idx,
         "path_to_data_dir_1": path_to_data_dir_1,
         "path_to_ml_training_dataset": path_to_ml_training_dataset}
    
    return converted_cmd_line_args



###########################
## Define error messages ##
###########################

_parse_and_convert_cmd_line_args_err_msg_1 = \
    ("The correct form of the command is:\n"
     "\n"
     "    python execute_main_action_steps.py "
     "--ml_model_task=<ml_model_task> "
     "--ml_model_idx=<ml_model_idx> "
     "--data_dir_1=<data_dir_1> "
     "--ml_training_dataset=<ml_training_dataset>\n"
     "\n"
     "where ``<ml_model_task>`` must be {}; ``<ml_model_idx>`` must be a "
     "nonnegative integer; ``<data_dir_1>`` must be the absolute path to a "
     "valid directory; and ``<ml_training_dataset>`` must be the absolute path "
     "to a valid directory.")



#########################
## Main body of script ##
#########################

# Parse the command line arguments.
converted_cmd_line_args = \
    parse_and_convert_cmd_line_args()
ml_model_task = \
    converted_cmd_line_args["ml_model_task"]
ml_model_idx = \
    converted_cmd_line_args["ml_model_idx"]
path_to_data_dir_1 = \
    converted_cmd_line_args["path_to_data_dir_1"]
path_to_ml_training_dataset = \
    converted_cmd_line_args["path_to_ml_training_dataset"]



# Set a RNG seed for reproducibility.
rng_seed = 20000 + ml_model_idx
    
torch.manual_seed(seed=rng_seed)
torch.cuda.manual_seed_all(seed=rng_seed)
random.seed(a=rng_seed)
np.random.seed(seed=rng_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Select the ``emicroml`` submodule required to generate a ML dataset that is
# appropriate to the specified ML model task. 
module_name = "emicroml.modelling.{}".format(ml_model_task).replace("/", ".")
ml_model_task_module = sys.modules[module_name]



# Load the training and validation ML datasets.
ml_dataset_types = ("training", "validation")
unformatted_path = (str(pathlib.Path(path_to_ml_training_dataset).parent)
                    + "/ml_dataset_for_{}.h5")
for ml_dataset_type in ml_dataset_types:
    path_to_ml_dataset = unformatted_path.format(ml_dataset_type)

    kwargs = {"path_to_ml_dataset": path_to_ml_dataset,
              "entire_ml_dataset_is_to_be_cached": False,
              "ml_data_values_are_to_be_checked": True,
              "max_num_ml_data_instances_per_chunk": 32}
    ml_dataset = ml_model_task_module.MLDataset(**kwargs)

    if ml_dataset_type == "training":
        ml_training_dataset = ml_dataset
    else:
        ml_validation_dataset = ml_dataset



# Select the ``emicroml`` submodule required to train a ML model that is
# appropriate to the specified ML model task. Also, select various ML model
# training parameters according to the specified ML model index and ML model
# task.
if ml_model_task == "cbed/distortion/estimation":
    architecture_set = ("distoptica_net",)

    attr_name = "num_pixels_across_each_cbed_pattern"
    num_pixels_across_each_cbed_pattern = getattr(ml_training_dataset,
                                                  attr_name)

    mini_batch_size_set = (64,)

    num_epochs_during_warmup_set = (4,)
    initial_lr_set = (1e-8,)
    max_lr_set = (5e-3,)

    weight_decay_set = (7.25e-4,)
    momentum_factor_set = (0.9,)
    
    min_lr_in_first_annealing_cycle_set = (2e-5,)
    num_lr_annealing_cycles_set = (1,)
    num_epochs_in_first_lr_annealing_cycle_set = (16,)
    multiplicative_decay_factor_set = (0.5,)
elif ml_model_task == "cbed/disk/localization":
    architecture_set = 5*("cbedd_loc_net",)

    attr_name = "num_pixels_across_each_cropped_cbed_pattern"
    num_pixels_across_each_cropped_cbed_pattern = getattr(ml_training_dataset,
                                                          attr_name)

    mini_batch_size_set = len(architecture_set)*(64,)
    
    num_epochs_during_warmup_set = len(architecture_set)*(16,)
    initial_lr_set = len(architecture_set)*(1e-8,)
    max_lr_set = len(architecture_set)*(2.56e-1,)

    weight_decay_set = len(architecture_set)*(10**(-4.5),)
    momentum_factor_set = len(architecture_set)*(0.9,)
    
    min_lr_in_first_annealing_cycle_set = len(architecture_set)*(4.68e-2,)
    num_lr_annealing_cycles_set = len(architecture_set)*(1,)
    num_epochs_in_first_lr_annealing_cycle_set = len(architecture_set)*(46,)
    multiplicative_decay_factor_set = len(architecture_set)*(0.5,)

    bce_loss_weight_set = (0.50, 0.75, 1.00, 1.25, 1.50)
elif ml_model_task == "cbed/disk/segmentation":
    architecture_set = 5*("cbedd_seg_net",)

    attr_name = "num_pixels_across_each_cropped_cbed_pattern"
    num_pixels_across_each_cropped_cbed_pattern = getattr(ml_training_dataset,
                                                          attr_name)

    wavelet_name_set = len(architecture_set)*("db8",)
    j_dashv_minus_j_epsilon_set = len(architecture_set)*(3,)

    mini_batch_size_set = len(architecture_set)*(64,)

    num_epochs_during_warmup_set = len(architecture_set)*(16,)
    initial_lr_set = len(architecture_set)*(1e-8,)
    max_lr_set = len(architecture_set)*(2.56e-1,)

    weight_decay_set = len(architecture_set)*(10**(-4.5),)
    momentum_factor_set = len(architecture_set)*(0.9,)

    min_lr_in_first_annealing_cycle_set = len(architecture_set)*(4.68e-2,)
    num_lr_annealing_cycles_set = len(architecture_set)*(1,)
    num_epochs_in_first_lr_annealing_cycle_set = len(architecture_set)*(46,)
    multiplicative_decay_factor_set = len(architecture_set)*(0.5,)



# Construct the ML dataset manager.
M = len(mini_batch_size_set)
mini_batch_size = mini_batch_size_set[ml_model_idx%M]

kwargs = {"ml_training_dataset": ml_training_dataset,
          "ml_validation_dataset": ml_validation_dataset,
          "mini_batch_size": mini_batch_size,
          "rng_seed": rng_seed,
          "num_data_loader_workers": 32}
ml_dataset_manager = ml_model_task_module.MLDatasetManager(**kwargs)



# Construct the ML optimizer.
ml_optimizer_params = {"base_lr": max_lr_set[ml_model_idx%M],
                       "weight_decay": weight_decay_set[ml_model_idx%M],
                       "momentum_factor": momentum_factor_set[ml_model_idx%M]}

kwargs = {"ml_optimizer_name": "sgd",
          "ml_optimizer_params": ml_optimizer_params}
ml_optimizer = emicroml.modelling.optimizers.Generic(**kwargs)



# Define the complete learning rate (LR) scheduler used throughout the ML model
# training and construct the LR scheduler manager. The complete LR scheduler
# starts with a linear warmup, followed by a cosine annealing cycle.
num_training_ml_data_instances = \
    len(ml_training_dataset)
num_training_mini_batch_instances_per_epoch = \
    ((num_training_ml_data_instances//mini_batch_size)
     + ((num_training_ml_data_instances%mini_batch_size) != 0))

num_epochs_during_warmup = num_epochs_during_warmup_set[ml_model_idx%M]

total_num_steps_in_lr_warmup = ((num_epochs_during_warmup
                                 * num_training_mini_batch_instances_per_epoch)
                                - 1)

start_scale_factor = initial_lr_set[ml_model_idx%M]/max_lr_set[ml_model_idx%M]

lr_scheduler_params = {"ml_optimizer": ml_optimizer,
                       "total_num_steps": total_num_steps_in_lr_warmup,
                       "start_scale_factor": start_scale_factor,
                       "end_scale_factor": 1.0}

kwargs = \
    {"lr_scheduler_name": "linear",
     "lr_scheduler_params": lr_scheduler_params}
non_sequential_lr_scheduler_1 = \
    emicroml.modelling.lr.schedulers.Nonsequential(**kwargs)

T = (num_epochs_in_first_lr_annealing_cycle_set[ml_model_idx%M]
     * num_training_mini_batch_instances_per_epoch)
num_steps_in_first_lr_annealing_cycle = T
num_lr_annealing_cycles = num_lr_annealing_cycles_set[ml_model_idx%M]
cycle_period_scale_factor = 2
multiplicative_decay_factor = multiplicative_decay_factor_set[ml_model_idx%M]

total_num_steps_in_lr_annealing_schedule = \
    sum(T * (cycle_period_scale_factor**cycle_idx)
        for cycle_idx
        in range(num_lr_annealing_cycles))

min_lr_in_first_annealing_cycle = \
    min_lr_in_first_annealing_cycle_set[ml_model_idx%M]

lr_scheduler_params = {"ml_optimizer": \
                       ml_optimizer,
                       "total_num_steps": \
                       total_num_steps_in_lr_annealing_schedule,
                       "num_steps_in_first_cycle": \
                       num_steps_in_first_lr_annealing_cycle,
                       "cycle_period_scale_factor": \
                       cycle_period_scale_factor,
                       "min_lr_in_first_cycle": \
                       min_lr_in_first_annealing_cycle,
                       "multiplicative_decay_factor": \
                       multiplicative_decay_factor}

lr_scheduler_name = "cosine_annealing_with_warm_restarts"

kwargs = \
    {"lr_scheduler_name": lr_scheduler_name,
     "lr_scheduler_params": lr_scheduler_params}
non_sequential_lr_scheduler_2 = \
    emicroml.modelling.lr.schedulers.Nonsequential(**kwargs)

non_sequential_lr_schedulers = (non_sequential_lr_scheduler_1,
                                non_sequential_lr_scheduler_2)

lr_scheduler_params = {"non_sequential_lr_schedulers": \
                       non_sequential_lr_schedulers}
kwargs = {"lr_scheduler_name": "sequential",
          "lr_scheduler_params": lr_scheduler_params}
generic_lr_scheduler = emicroml.modelling.lr.schedulers.Generic(**kwargs)

kwargs = {"lr_schedulers": (generic_lr_scheduler,),
          "phase_in_which_to_update_lr": "training"}
lr_scheduler_manager = emicroml.modelling.lr.LRSchedulerManager(**kwargs)



# Construct the ML model trainer.
partial_path_1 = pathlib.Path(path_to_ml_training_dataset).parent.name
partial_path_2 = (partial_path_1.replace("ml_datasets_with", "/ml_models_for")
                  if ("ml_datasets_with" in partial_path_1)
                  else "")
unformatted_path = (path_to_data_dir_1
                    + "/ml_models"
                    + partial_path_2
                    + "/ml_model_{}")
output_dirname = unformatted_path.format(ml_model_idx)

misc_model_training_metadata = {"ml_model_architecture": \
                                architecture_set[ml_model_idx%M],
                                "rng_seed": \
                                rng_seed}

kwargs = {"ml_dataset_manager": ml_dataset_manager,
          "device_name": None,
          "checkpoints": None,
          "lr_scheduler_manager": lr_scheduler_manager,
          "output_dirname": output_dirname,
          "misc_model_training_metadata": misc_model_training_metadata}
ml_model_trainer = ml_model_task_module.MLModelTrainer(**kwargs)



# Initialize the ML model.
if ml_model_task == "cbed/disk/segmentation":
    kwargs = {"single_dim_slice": slice(0, 1), "device_name": "cpu"}
    ml_data_instances = ml_dataset.get_ml_data_instances(**kwargs)
    
    disk_boundary_sample_size = \
        ml_data_instances["principal_disk_boundary_pt_sets"].shape[1]
    j_dashv = \
        round(np.log2(disk_boundary_sample_size))
    j_epsilon = \
        j_dashv-j_dashv_minus_j_epsilon_set[ml_model_idx%M]

ml_model_ctor_params = \
    {"mini_batch_norm_eps": \
     1e-5,
     "normalization_weights": \
     ml_training_dataset.normalization_weights,
     "normalization_biases": \
     ml_training_dataset.normalization_biases}
if ml_model_task == "cbed/distortion/estimation":
    ml_model_ctor_params = \
        {**ml_model_ctor_params,
         "num_pixels_across_each_cbed_pattern": \
         num_pixels_across_each_cbed_pattern,
         "architecture": \
         architecture_set[ml_model_idx%M]}
elif ml_model_task == "cbed/disk/localization":
    ml_model_ctor_params = \
        {**ml_model_ctor_params,
         "num_pixels_across_each_cropped_cbed_pattern": \
         num_pixels_across_each_cropped_cbed_pattern,
         "num_downsamplings": \
         6,
         "bce_loss_weight": \
         bce_loss_weight_set[ml_model_idx%M]}
elif ml_model_task == "cbed/disk/segmentation":
    ml_model_ctor_params = \
        {**ml_model_ctor_params,
         "num_pixels_across_each_cropped_cbed_pattern": \
         num_pixels_across_each_cropped_cbed_pattern,
         "wavelet_name": \
         wavelet_name_set[ml_model_idx%M],
         "j_epsilon": \
         j_epsilon,
         "j_dashv": \
         j_dashv}

kwargs = ml_model_ctor_params
ml_model = ml_model_task_module.MLModel(**kwargs)



# Train the ML model.
ml_model_param_groups = (ml_model.parameters(),)

ml_model_trainer.train_ml_model(ml_model, ml_model_param_groups)



# Calculate the precision-recall (PR) curve if the ML model is performing binary
# classification; save the PR curve to file; estimate the decision threshold
# that yields the maximum precision for a recall greater than or equal to 0.8;
# and update the trained ML model with this new decision threshold.
if ml_model_task == "cbed/disk/localization":
    path_to_ml_model_training_summary_output_data = \
        output_dirname + "/ml_model_training_summary_output_data.h5"

    unformatted_msg = ("Calculating precision-recall (PR) curve using the "
                       "ML model training summary output data stored in the "
                       "file ``'{}'``...\n")
    msg = unformatted_msg.format(path_to_ml_model_training_summary_output_data)
    print(msg)

    path_to_pr_curve_data = output_dirname + "/pr_curve_data.h5"

    kwargs = \
        {"path_to_ml_model_training_summary_output_data": \
         path_to_ml_model_training_summary_output_data,
         "single_dim_slice": \
         slice(-len(ml_validation_dataset), None),
         "path_to_pr_curve_data": \
         path_to_pr_curve_data}
    precisions, recalls, decision_thresholds = \
        ml_model_task_module.calc_and_save_pr_curve(**kwargs)

    unformatted_msg = ("Finished calculating the PR curve. The PR curve data "
                       "has been saved to the file ``'{}'``.\n")
    msg = unformatted_msg.format(path_to_pr_curve_data)
    print(msg)



    min_acceptable_recall = 0.8
    mask = (recalls >= min_acceptable_recall)
    
    if not torch.any(mask):
        unformatted_err_msg = ("No decision threshold achieves a recall >= "
                               "{}.\n\n\n")
        err_msg = unformatted_err_msg.format(min_acceptable_recall)
        print(err_msg)
    else:
        decision_threshold_idx = torch.argmax(precisions[mask])

        precision = precisions[decision_threshold_idx].item()
        recall = recalls[decision_threshold_idx].item()
        decision_threshold = decision_thresholds[decision_threshold_idx].item()

        ml_model.update_decision_threshold(decision_threshold)

        pattern = r"ml_model_at_lr_step_[0-9]*\.pth"
        largest_lr_step_idx = max([name.split("_")[-1].split(".")[0]
                                   for name in os.listdir(output_dirname)
                                   if re.fullmatch(pattern, name)])

        unformatted_filename = \
            output_dirname + "/ml_model_at_lr_step_{}.pth"
        ml_model_state_dict_filename = \
            unformatted_filename.format(largest_lr_step_idx)

        kwargs = {"obj": ml_model.state_dict(),
                  "f": ml_model_state_dict_filename}
        torch.save(**kwargs)

        unformatted_msg = ("Updated the decision threshold of the trained ML "
                           "model to be ``{}``, corresponding to a precision "
                           "of ``{}`` and a recall of ``{}``.\n\n\n")
        msg = unformatted_msg.format(decision_threshold, precision, recall)
        print(msg)
