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

# For creating path objects.
import pathlib

# For removing directories.
import shutil



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
argument_names = ("ml_model_task", "data_dir_1")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
ml_model_task = args.ml_model_task
path_to_data_dir_1 = args.data_dir_1

sample_name = "MoS2_on_amorphous_C"

unformatted_path = (path_to_data_dir_1
                    + "/ml_datasets"
                    + "/ml_datasets_for_ml_model_test_set_1"
                    + "/ml_datasets_with_cbed_patterns_of_{}")
path_to_input_ml_datasets = unformatted_path.format(sample_name)

pattern = "ml_datasets_with_[a-z]*_sized_disks"
partial_path_set_1 = [path_to_input_ml_datasets + "/" + name
                      for name in os.listdir(path_to_input_ml_datasets)
                      if re.fullmatch(pattern, name)]
for partial_path_1 in partial_path_set_1:
    pattern = "ml_dataset_[0-9]*\.h5"
    input_ml_dataset_filenames = [partial_path_1 + "/" + name
                                  for name in os.listdir(partial_path_1)
                                  if re.fullmatch(pattern, name)]

    output_ml_dataset_basename = \
        pathlib.Path(partial_path_1).name.replace("datasets", "dataset")
    output_ml_dataset_filename = \
        path_to_input_ml_datasets + "/" + output_ml_dataset_basename + ".h5"

    module_alias = emnn.modelling.cbed.distortion.estimation

    kwargs = {"input_ml_dataset_filenames": input_ml_dataset_filenames,
              "output_ml_dataset_filename": output_ml_dataset_filename,
              "rm_input_ml_dataset_files": True,
              "max_num_ml_data_instances_per_file_update": 240}
    module_alias.combine_ml_dataset_files(**kwargs)

    shutil.rmtree(partial_path_1)
