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
import emnn.modelling.optimizers
import emnn.modelling.lr.schedulers
import emnn.modelling.cbed.distortion.estimation



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2023"
__credits__      = ["Matthew Fitzpatrick"]
__maintainer__   = "Matthew Fitzpatrick"
__email__        = "mrfitzpa@uvic.ca"
__status__       = "Development"



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
argument_names = ("ml_model_task", "test_idx", "data_dir_1")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
test_idx = int(args.test_idx)
path_to_data_dir_1 = args.data_dir_1



if ml_model_task == "cbed/fg/segmentation":
    module_alias_1 = emnn.modelling.cbed.fg.segmentation
elif ml_model_task == "cbed/disk/segmentation":
    module_alias_1 = emnn.modelling.cbed.disk.segmentation
elif ml_model_task == "cbed/distortion/estimation":
    module_alias_1 = emnn.modelling.cbed.distortion.estimation
    architecture = "multi_head_mod_resnet_57"



device_name = None


    
path_to_ml_dataset = (path_to_data_dir_1
                      + "/ml_datasets/ml_dataset_for_training.h5")
kwargs = {"path_to_ml_dataset": path_to_ml_dataset,
          "entire_ml_dataset_is_to_be_cached": True,
          "ml_data_values_are_to_be_checked": True,
          "max_num_ml_data_instances_per_chunk": 32,
          "auxiliary_device_name": device_name}
ml_training_dataset = module_alias_1.MLDataset(**kwargs)
ml_training_dataset.disk_overlap_class_weights



mini_batch_size = 64

kwargs = {"ml_training_dataset": ml_training_dataset,
          "ml_validation_dataset": None,
          "mini_batch_size": mini_batch_size}
ml_dataset_manager = module_alias_1.MLDatasetManager(**kwargs)



num_epochs = 1

T = num_epochs
B = len(ml_training_dataset)
b = mini_batch_size

lambda_norm_set = (1e-4, 1e-3, 1e-2, 1e-1)
lambda_norm = lambda_norm_set[test_idx%len(lambda_norm_set)]

weight_decay = lambda_norm * ((b / B / T)**0.5)



checkpoints = tuple()



base_lr = 1e-5



module_alias_2 = emnn.modelling.optimizers

ml_optimizer_params = {"base_lr": base_lr, "weight_decay": weight_decay}

kwargs = {"ml_optimizer_name": "adam_w",
          "ml_optimizer_params": ml_optimizer_params}
ml_optimizer = module_alias_2.Generic(**kwargs)



module_alias_3 = emnn.modelling.lr.schedulers

num_ml_data_instances = len(ml_training_dataset)
num_mini_batches_per_epoch = ((num_ml_data_instances//mini_batch_size)
                              + ((num_ml_data_instances%mini_batch_size) != 0))
total_num_mini_batches = num_mini_batches_per_epoch*num_epochs

alpha_initial = -np.log10(base_lr)
alpha_final = 2
tau = -(total_num_mini_batches-1) / (alpha_final-alpha_initial)

lr_scheduler_params = {"ml_optimizer": ml_optimizer,
                       "total_num_steps": total_num_mini_batches-1,
                       "multiplicative_factor": 10**(1/tau)}
kwargs = {"lr_scheduler_name": "exponential",
          "lr_scheduler_params": lr_scheduler_params}
lr_scheduler = module_alias_3.Generic(**kwargs)

module_alias_4 = emnn.modelling.lr
kwargs = {"lr_schedulers": (lr_scheduler,),
          "phase_in_which_to_update_lr": "training"}
lr_scheduler_manager = module_alias_4.LRSchedulerManager(**kwargs)



unformatted_path = (path_to_data_dir_1
                    + "/learning_rate_range_test_results"
                    + "/test_{}/using_{}_architecture")
args = (test_idx, architecture)
output_dirname = unformatted_path.format(*args)



ml_model_ctor_params = {"num_pixels_across_each_cbed_pattern": \
                        ml_training_dataset.num_pixels_across_each_cbed_pattern,
                        "mini_batch_norm_eps": \
                        1e-5}

if ml_model_task == "cbed/fg/segmentation":
    pass

elif ml_model_task == "cbed/disk/segmentation":
    pass
    
elif ml_model_task == "cbed/distortion/estimation":
    ml_model_ctor_params["architecture"] = \
        architecture
    ml_model_ctor_params["max_num_disks_in_any_cbed_pattern"] = \
        ml_training_dataset.max_num_disks_in_any_cbed_pattern
    ml_model_ctor_params["normalization_weights"] = \
        ml_training_dataset.normalization_weights
    ml_model_ctor_params["normalization_biases"] = \
        ml_training_dataset.normalization_biases

    rng_seed = 50200 + 1000*test_idx



misc_model_training_metadata = {"ml_model_architecture": architecture,
                                "rng_seed": rng_seed}



kwargs = {"ml_dataset_manager": ml_dataset_manager,
          "device_name": device_name,
          "checkpoints": checkpoints,
          "lr_scheduler_manager": lr_scheduler_manager,
          "output_dirname": output_dirname,
          "misc_model_training_metadata": misc_model_training_metadata}
ml_model_trainer = module_alias_1.MLModelTrainer(**kwargs)


    
torch.manual_seed(seed=rng_seed)

kwargs = ml_model_ctor_params
ml_model = module_alias_1.MLModel(**kwargs)

ml_model_param_groups = (ml_model.parameters(),)



ml_model_trainer.train_ml_model(ml_model, ml_model_param_groups)
