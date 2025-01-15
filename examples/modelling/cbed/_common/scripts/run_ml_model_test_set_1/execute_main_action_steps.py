"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For listing files and subdirectories in a given directory, and for renaming
# directories.
import os

# For pattern matching.
import re



# For avoiding errors related to the ``mkl-service`` package. Note that
# ``numpy`` needs to be imported before ``torch``.
import numpy as np

# For setting the seed to the random-number-generator used in ``pytorch``.
import torch



# For combining datasets.
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



###############################################
## Define classes, functions, and contstants ##
###############################################



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

parser = argparse.ArgumentParser()
argument_names = ("ml_model_task", "data_dir_1")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
path_to_data_dir_1 = args.data_dir_1



rng_seed = 555444
torch.manual_seed(seed=rng_seed)



mini_batch_size = 64
device_name = None



path_to_ml_models = path_to_data_dir_1 + "/ml_models"
pattern = "ml_model_[0-9]*"
ml_model_idx_set = tuple(int(name.split("_")[-1])
                         for name in os.listdir(path_to_ml_models)
                         if re.fullmatch(pattern, name))



sample_name = "MoS2_on_amorphous_C"

unformatted_path = (path_to_data_dir_1
                    + "/ml_datasets"
                    + "/ml_datasets_for_ml_model_test_set_1"
                    + "/ml_datasets_with_cbed_patterns_of_{}")
path_to_ml_datasets = unformatted_path.format(sample_name)

pattern = "ml_dataset_with_[a-z]*_sized_disks\.h5"
disk_sizes = tuple(name.split("_")[-3]
                   for name in os.listdir(path_to_ml_datasets)
                   if re.fullmatch(pattern, name))



for disk_size in disk_sizes:
    unformatted_path = (path_to_data_dir_1
                        + "/ml_datasets"
                        + "/ml_datasets_for_ml_model_test_set_1"
                        + "/ml_datasets_with_cbed_patterns_of_{}"
                        + "/ml_dataset_with_{}_sized_disks.h5")
    path_to_ml_dataset = unformatted_path.format(sample_name, disk_size)

    module_alias = emnn.modelling.cbed.distortion.estimation
    kwargs = {"path_to_ml_dataset": path_to_ml_dataset,
              "entire_ml_dataset_is_to_be_cached": True,
              "ml_data_values_are_to_be_checked": True,
              "max_num_ml_data_instances_per_chunk": 32,
              "auxiliary_device_name": device_name}
    ml_testing_dataset = module_alias.MLDataset(**kwargs)
    ml_testing_dataset.disk_overlap_class_weights

    kwargs = {"ml_training_dataset": None,
              "ml_validation_dataset": None,
              "ml_testing_dataset": ml_testing_dataset,
              "mini_batch_size": mini_batch_size}
    ml_dataset_manager = module_alias.MLDatasetManager(**kwargs)


    
    for ml_model_idx in ml_model_idx_set:
        architecture = "multi_head_mod_resnet_57"

        unformatted_path = \
            (path_to_data_dir_1
             + "/ml_models/ml_model_{}")
        path_to_ml_model_state_dicts = \
            unformatted_path.format(ml_model_idx, architecture)
        pattern = \
            "ml_model_at_lr_step_[0-9]*\.pth"
        largest_lr_step_idx = \
            max([name.split("_")[-1].split(".")[0]
                 for name in os.listdir(path_to_ml_model_state_dicts)
                 if re.fullmatch(pattern, name)])

        

        unformatted_filename = \
            (path_to_ml_model_state_dicts + "/ml_model_at_lr_step_{}.pth")
        ml_model_state_dict_filename = \
            unformatted_filename.format(largest_lr_step_idx)

        kwargs = {"ml_model_state_dict_filename": ml_model_state_dict_filename,
                  "device_name": device_name}
        ml_model = module_alias.load_ml_model_from_file(**kwargs)



        unformatted_path = (path_to_ml_model_state_dicts
                            + "/ml_model_test_set_1_results"
                            + "/results_for_cbed_patterns_of_{}"
                            + "_with_{}_sized_disks")
        output_dirname = unformatted_path.format(sample_name, disk_size)

        

        misc_model_testing_metadata = {"ml_model_architecture": architecture}

        

        kwargs = {"ml_dataset_manager": ml_dataset_manager,
                  "device_name": device_name,
                  "output_dirname": output_dirname,
                  "misc_model_testing_metadata": misc_model_testing_metadata}
        ml_model_tester = module_alias.MLModelTester(**kwargs)

        ml_model_tester.test_ml_model(ml_model)
