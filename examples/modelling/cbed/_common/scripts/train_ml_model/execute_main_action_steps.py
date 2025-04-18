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

# For setting Python's seed.
import random



# For avoiding errors related to the ``mkl-service`` package. Note that
# ``numpy`` needs to be imported before ``torch``.
import numpy as np

# For setting the seed to the random-number-generator used in ``pytorch``.
import torch



# For training models.
import emicroml.modelling.optimizers
import emicroml.modelling.lr.schedulers
import emicroml.modelling.cbed.distortion.estimation



##############################################
## Define classes, functions, and constants ##
##############################################

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
    architecture_set = ("distoptica_net",)
    # architecture_set = ("no_pool_resnet_39",)
    # rng_seed = 18200 + 1000*ml_model_idx
    rng_seed = 20000



torch.manual_seed(seed=rng_seed)
torch.cuda.manual_seed_all(seed=rng_seed)
random.seed(a=rng_seed)
np.random.seed(seed=rng_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



mini_batch_size_set = (64,)

weight_decay_set = (1.5e-2,)
max_lr_set = (5e-3,)
initial_lr_set = (1e-8,)*len(max_lr_set)
reduction_factor_set = (1/2,)*len(max_lr_set)
min_lr_in_first_annealing_cycle_set = (1e-8,)*len(max_lr_set)
num_lr_annealing_cycles_set = (1,)*len(max_lr_set)
num_epochs_in_first_lr_annealing_cycle_set = (40,)*len(max_lr_set)
multiplicative_decay_factor_set = (0.5,)*len(max_lr_set)
num_epochs_during_warmup_set = (5,)*len(max_lr_set)
num_epochs_after_warmup_set = (80,)*len(max_lr_set)

M_1 = len(architecture_set)
M_2 = len(max_lr_set)

momentum_factor = 0.9

mini_batch_size = mini_batch_size_set[ml_model_idx%M_2]
reduction_factor = reduction_factor_set[ml_model_idx%M_2]
num_epochs_during_warmup = num_epochs_during_warmup_set[ml_model_idx%M_2]
num_epochs_after_warmup = num_epochs_after_warmup_set[ml_model_idx%M_2]

# num_epochs_during_warmup = int(round(40 * (mini_batch_size/64)))

# num_epochs_after_warmup = int(round(90 * (mini_batch_size/64)))
# num_epochs_during_warmup = num_epochs_during_warmup_set[ml_model_idx%M_2]
# num_epochs_after_warmup = num_epochs_after_warmup_set[ml_model_idx%M_2]

# num_epochs_after_warmup = 60
# lambda_norm_set = (10**(-5), 10**(-4.5), 10**(-4))
# max_lr_set = (1e-2, 1e-2, 1e-2)

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

kwargs = {"ml_training_dataset": ml_training_dataset,
          "ml_validation_dataset": ml_validation_dataset,
          "mini_batch_size": mini_batch_size,
          "rng_seed": rng_seed}
ml_dataset_manager = module_alias_1.MLDatasetManager(**kwargs)

# checkpoints = None
checkpoints = tuple()



max_lr = \
    max_lr_set[ml_model_idx%M_2]
initial_lr = \
    initial_lr_set[ml_model_idx%M_2]
weight_decay = \
    weight_decay_set[ml_model_idx%M_2]
min_lr_in_first_annealing_cycle = \
    min_lr_in_first_annealing_cycle_set[ml_model_idx%M_2]

# T = 20
# B = len(ml_training_dataset)
# b = mini_batch_size

# lambda_norm = lambda_norm_set[ml_model_idx%M_2]

# weight_decay = lambda_norm * ((b / B / T)**0.5)




module_alias_2 = emicroml.modelling.optimizers

ml_optimizer_params = {"base_lr": max_lr,
                       "weight_decay": weight_decay,
                       "momentum_factor": momentum_factor}

kwargs = {"ml_optimizer_name": "sgd",
          "ml_optimizer_params": ml_optimizer_params}
ml_optimizer = module_alias_2.Generic(**kwargs)

# ml_optimizer_params = {"base_lr": max_lr, "weight_decay": weight_decay}

# kwargs = {"ml_optimizer_name": "adam_w",
#           "ml_optimizer_params": ml_optimizer_params}
# ml_optimizer = module_alias_2.Generic(**kwargs)



module_alias_3 = emicroml.modelling.lr.schedulers

num_training_ml_data_instances = \
    len(ml_training_dataset)
num_training_mini_batch_instances_per_epoch = \
    ((num_training_ml_data_instances//mini_batch_size)
     + ((num_training_ml_data_instances%mini_batch_size) != 0))

total_num_steps_in_lr_warmup = \
    ((num_epochs_during_warmup
      * num_training_mini_batch_instances_per_epoch)
     - 1)
# total_num_steps_in_lr_warmup = \
#     num_epochs_during_warmup - 1

lr_scheduler_params = {"ml_optimizer": ml_optimizer,
                       "total_num_steps": total_num_steps_in_lr_warmup,
                       "start_scale_factor": initial_lr/max_lr,
                       "end_scale_factor": 1.0}
kwargs = {"lr_scheduler_name": "linear",
          "lr_scheduler_params": lr_scheduler_params}
non_sequential_lr_scheduler_1 = module_alias_3.Nonsequential(**kwargs)

T = (num_epochs_in_first_lr_annealing_cycle_set[ml_model_idx%M_2]
     * num_training_mini_batch_instances_per_epoch)
# T = num_epochs_in_first_lr_annealing_cycle_set[ml_model_idx%M_2]
num_steps_in_first_lr_annealing_cycle = T
num_lr_annealing_cycles = num_lr_annealing_cycles_set[ml_model_idx%M_2]
cycle_period_scale_factor = 2
multiplicative_decay_factor = multiplicative_decay_factor_set[ml_model_idx%M_2]

# total_num_steps_in_lr_annealing_schedule = \
#     sum(T * (cycle_period_scale_factor**cycle_idx)
#         for cycle_idx
#         in range(num_lr_annealing_cycles))

total_num_steps_in_lr_annealing_schedule = \
    (num_epochs_after_warmup
     * num_training_mini_batch_instances_per_epoch)
num_steps_in_first_lr_annealing_cycle = \
    total_num_steps_in_lr_annealing_schedule

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

# num_validation_ml_data_instances = \
#     len(ml_validation_dataset)
# num_validation_mini_batch_instances_per_epoch = \
#     ((num_validation_ml_data_instances//mini_batch_size)
#      + ((num_validation_ml_data_instances%mini_batch_size) != 0))

# lr_scheduler_params = {"ml_optimizer": \
#                        ml_optimizer,
#                        "total_num_steps": \
#                        num_epochs_after_warmup,
#                        "reduction_factor": \
#                        0.5,
#                        "max_num_steps_of_stagnation": \
#                        5,
#                        "improvement_threshold": \
#                        0.01,
#                        "averaging_window_in_steps": \
#                        num_validation_mini_batch_instances_per_epoch}
# kwargs = {"lr_scheduler_name": "reduce_on_plateau",
#           "lr_scheduler_params": lr_scheduler_params}
# non_sequential_lr_scheduler_2 = module_alias_3.Nonsequential(**kwargs)

lr_scheduler_params = {"ml_optimizer": \
                       ml_optimizer,
                       "total_num_steps": \
                       total_num_steps_in_lr_annealing_schedule,
                       "reduction_factor": \
                       reduction_factor,
                       "max_num_steps_of_stagnation": \
                       3*num_training_mini_batch_instances_per_epoch,
                       "improvement_threshold": \
                       0.01,
                       "averaging_window_in_steps": \
                       num_training_mini_batch_instances_per_epoch}
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
          # "phase_in_which_to_update_lr": "validation"}
          "phase_in_which_to_update_lr": "training"}
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
