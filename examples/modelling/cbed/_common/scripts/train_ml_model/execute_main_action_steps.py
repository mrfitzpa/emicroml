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
"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse



# For avoiding errors related to the ``mkl-service`` package. Note that
# ``numpy`` needs to be imported before ``torch``.
import numpy as np

# For setting the seed to the random-number-generator used in ``pytorch``.
import torch



# For training models.
import emicroml.modelling.optimizers
import emicroml.modelling.lr.schedulers
import emicroml.modelling.cbed.distortion.estimation



###############################################
## Define classes, functions, and contstants ##
###############################################

def _parse_and_convert_cmd_line_args():
    accepted_ml_model_tasks = ("cbed/distortion/estimation",)

    current_func_name = "_parse_and_convert_cmd_line_args"

    try:
        parser = argparse.ArgumentParser()
        argument_names = ("ml_model_task", "ml_model_idx", "data_dir_1")
        for argument_name in argument_names:
            parser.add_argument("--"+argument_name)
        args = parser.parse_args()
        ml_model_task = args.ml_model_task
        ml_model_idx = int(args.ml_model_idx)
        path_to_data_dir_1 = args.data_dir_1

        if ((ml_model_task not in accepted_ml_model_tasks)
            or (ml_model_idx < 0)):
            raise
    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(accepted_ml_model_tasks[0])
        raise SystemExit(err_msg)

    converted_cmd_line_args = {"ml_model_task": ml_model_task,
                               "ml_model_idx": ml_model_idx,
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
     "--ml_model_idx=<ml_model_idx> "
     "--data_dir_1=<data_dir_1>\n"
     "\n"
     "where ``<ml_model_task>`` must be set to {}; ``<ml_model_idx>`` can be "
     "any nonnegative integer, and ``<data_dir_1>`` must be the absolute path "
     "to a valid directory.")



#########################
## Main body of script ##
#########################

converted_cmd_line_args = _parse_and_convert_cmd_line_args()
ml_model_task = converted_cmd_line_args["ml_model_task"]
ml_model_idx = converted_cmd_line_args["ml_model_idx"]
path_to_data_dir_1 = converted_cmd_line_args["path_to_data_dir_1"]



if ml_model_task == "cbed/distortion/estimation":
    module_alias_1 = emicroml.modelling.cbed.distortion.estimation
    architecture_set = ("no_pool_resnet_39",)
    rng_seed = 18200 + 1000*ml_model_idx



torch.manual_seed(seed=rng_seed)



mini_batch_size_set = (64,)



num_epochs_after_warmup = 60
lambda_norm_set = (10**(-5), 10**(-4.5), 10**(-4))
max_lr_set = (1e-2, 1e-2, 1e-2)

# num_steps_in_first_lr_annealing_cycle_set = (20,)
# lambda_norm_set = (5e-4, 5e-3, 5e-4, 5e-3)
# max_lr_set = (5e-3, 5e-3, 1e-3, 1e-3)
# min_lr_in_first_annealing_cycle_set = (1e-4, 1e-4, 5e-5, 5e-5)
# num_lr_annealing_cycles_set = (2, 2, 2, 2)
# multiplicative_decay_factor_set = (0.5, 0.5, 0.5, 0.5)

# num_steps_in_first_lr_annealing_cycle_set = (20,)
# lambda_norm_set = (5e-5,)
# max_lr_set = (5e-3,)
# min_lr_in_first_annealing_cycle_set = (1e-4,)
# num_lr_annealing_cycles_set = (2,)
# multiplicative_decay_factor_set = (0.5,)

M_1 = len(architecture_set)
M_2 = len(lambda_norm_set)
# M_2 = len(multiplicative_decay_factor_set)

# num_steps_in_first_lr_annealing_cycle = \
#     num_steps_in_first_lr_annealing_cycle_set[(ml_model_idx//M_2)%M_1]



device_name = None



ml_dataset_types = ("training", "validation")
for ml_dataset_type in ml_dataset_types:
    unformatted_path = path_to_data_dir_1 + "/ml_datasets/ml_dataset_for_{}.h5"
    path_to_ml_dataset = unformatted_path.format(ml_dataset_type)

    kwargs = {"path_to_ml_dataset": path_to_ml_dataset,
              "entire_ml_dataset_is_to_be_cached": True,
              "ml_data_values_are_to_be_checked": True,
              "max_num_ml_data_instances_per_chunk": 32}
    ml_dataset = module_alias_1.MLDataset(**kwargs)

    if ml_dataset_type == "training":
        ml_training_dataset = ml_dataset
    else:
        ml_validation_dataset = ml_dataset



architecture = architecture_set[(ml_model_idx//M_2)%M_1]
mini_batch_size = mini_batch_size_set[(ml_model_idx//M_2)%M_1]

kwargs = {"ml_training_dataset": ml_training_dataset,
          "ml_validation_dataset": ml_validation_dataset,
          "mini_batch_size": mini_batch_size}
ml_dataset_manager = module_alias_1.MLDatasetManager(**kwargs)

checkpoints = None



max_lr = \
    max_lr_set[ml_model_idx%M_2]
# min_lr_in_first_annealing_cycle = \
#     min_lr_in_first_annealing_cycle_set[ml_model_idx%M_2]

# T = num_steps_in_first_lr_annealing_cycle
T = 20
B = len(ml_training_dataset)
b = mini_batch_size

lambda_norm = lambda_norm_set[ml_model_idx%M_2]

weight_decay = lambda_norm * ((b / B / T)**0.5)



module_alias_2 = emicroml.modelling.optimizers

ml_optimizer_params = {"base_lr": max_lr, "weight_decay": weight_decay}

kwargs = {"ml_optimizer_name": "adam_w",
          "ml_optimizer_params": ml_optimizer_params}
ml_optimizer = module_alias_2.Generic(**kwargs)



module_alias_3 = emicroml.modelling.lr.schedulers

lr_scheduler_params = {"ml_optimizer": ml_optimizer,
                       "total_num_steps": 4,
                       "start_scale_factor": 1/mini_batch_size,
                       "end_scale_factor": 1.0}
kwargs = {"lr_scheduler_name": "linear",
          "lr_scheduler_params": lr_scheduler_params}
non_sequential_lr_scheduler_1 = module_alias_3.Nonsequential(**kwargs)

# num_lr_annealing_cycles = num_lr_annealing_cycles_set[ml_model_idx%M_2]
# cycle_period_scale_factor = 2
# multiplicative_decay_factor = multiplicative_decay_factor_set[ml_model_idx%M_2]

# total_num_steps_in_lr_annealing_schedule = \
#     sum(T * (cycle_period_scale_factor**cycle_idx)
#         for cycle_idx
#         in range(num_lr_annealing_cycles))

# lr_scheduler_params = {"ml_optimizer": \
#                        ml_optimizer,
#                        "total_num_steps": \
#                        total_num_steps_in_lr_annealing_schedule,
#                        "num_steps_in_first_cycle": \
#                        num_steps_in_first_lr_annealing_cycle,
#                        "cycle_period_scale_factor": \
#                        cycle_period_scale_factor,
#                        "min_lr_in_first_cycle": \
#                        min_lr_in_first_annealing_cycle,
#                        "multiplicative_decay_factor": \
#                        multiplicative_decay_factor}
# kwargs = {"lr_scheduler_name": "cosine_annealing_with_warm_restarts",
#           "lr_scheduler_params": lr_scheduler_params}
# non_sequential_lr_scheduler_2 = module_alias_3.Nonsequential(**kwargs)

num_validation_ml_data_instances = \
    len(ml_validation_dataset)
num_validation_mini_batch_instances_per_epoch = \
    ((num_validation_ml_data_instances//mini_batch_size)
     + ((num_validation_ml_data_instances%mini_batch_size) != 0))

lr_scheduler_params = {"ml_optimizer": \
                       ml_optimizer,
                       "total_num_steps": \
                       num_epochs_after_warmup,
                       "reduction_factor": \
                       0.5,
                       "max_num_steps_of_stagnation": \
                       5,
                       "improvement_threshold": \
                       0.01,
                       "averaging_window_in_steps": \
                       num_validation_mini_batch_instances_per_epoch}
kwargs = {"lr_scheduler_name": "reduce_on_plateau",
          "lr_scheduler_params": lr_scheduler_params}
non_sequential_lr_scheduler_2 = module_alias_3.Nonsequential(**kwargs)

non_sequential_lr_schedulers = (non_sequential_lr_scheduler_1,
                                non_sequential_lr_scheduler_2)

lr_scheduler_params = {"non_sequential_lr_schedulers": \
                       non_sequential_lr_schedulers}
kwargs = {"lr_scheduler_name": "sequential",
          "lr_scheduler_params": lr_scheduler_params}
generic_lr_scheduler = module_alias_3.Generic(**kwargs)

module_alias_4 = emicroml.modelling.lr
kwargs = {"lr_schedulers": (generic_lr_scheduler,),
          "phase_in_which_to_update_lr": "validation"}
lr_scheduler_manager = module_alias_4.LRSchedulerManager(**kwargs)



unformatted_path = (path_to_data_dir_1
                    + "/ml_models/ml_model_{}")
output_dirname = unformatted_path.format(ml_model_idx)



ml_model_ctor_params = {"num_pixels_across_each_cbed_pattern": \
                        ml_training_dataset.num_pixels_across_each_cbed_pattern,
                        "mini_batch_norm_eps": \
                        1e-5}

if ml_model_task == "cbed/distortion/estimation":
    ml_model_ctor_params["architecture"] = \
        architecture
    ml_model_ctor_params["normalization_weights"] = \
        ml_training_dataset.normalization_weights
    ml_model_ctor_params["normalization_biases"] = \
        ml_training_dataset.normalization_biases



misc_model_training_metadata = {"ml_model_architecture": architecture,
                                "rng_seed": rng_seed}

kwargs = {"ml_dataset_manager": ml_dataset_manager,
          "device_name": device_name,
          "checkpoints": checkpoints,
          "lr_scheduler_manager": lr_scheduler_manager,
          "output_dirname": output_dirname,
          "misc_model_training_metadata": misc_model_training_metadata}
ml_model_trainer = module_alias_1.MLModelTrainer(**kwargs)

kwargs = ml_model_ctor_params
ml_model = module_alias_1.MLModel(**kwargs)

ml_model_param_groups = (ml_model.parameters(),)

ml_model_trainer.train_ml_model(ml_model, ml_model_param_groups)
