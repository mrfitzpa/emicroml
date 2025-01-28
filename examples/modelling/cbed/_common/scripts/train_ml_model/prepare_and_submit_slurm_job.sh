#!/bin/bash
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



# The current script is expected to be called only by the parent script with the
# basename ``execute_all_action_steps.py``, located in the same directory as the
# current script. The parent script performs the "action" of ???
#
# The current script prepares and submits a SLURM job which:
#
#   1. Sets the remaining parameters required to execute the "main" steps of the
#   action that were not set by the parent script.
#
#   2. Prepares the input data required to execute the main steps.
#
#   3. Executes the main steps.
#
#   4. Moves non-temporary output data that are generated from the main steps to
#   their expected final destinations.
#
#   5. Deletes/removes any remaining temporary files or directories.
#
# The main steps are executed by running the script with the basename
# ``execute_main_action_steps.py``, located in the same directory as the current
# script.



#SBATCH --job-name=train_ml_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=46G               # memory per node
#SBATCH --time=00-23:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL



path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
ml_model_task=${4}
ml_model_idx=${5}
overwrite_slurm_tmpdir=${6}

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    SLURM_TMPDIR=${path_to_data_dir_1}
fi

basename=custom_env_setup_for_slurm_jobs.sh
if [ ! -f ${path_to_repo_root}/${basename} ]
then
    basename=default_env_setup_for_slurm_jobs.sh
fi
source ${path_to_repo_root}/${basename} ${SLURM_TMPDIR}/tempenv false

partial_path_1=ml_datasets
partial_path_2=${partial_path_1}/ml_datasets_for_training_and_validation

ml_dataset_types=(training validation)

for ml_dataset_type in "${ml_dataset_types[@]}"
do
    dirname_1=${path_to_data_dir_1}/${partial_path_1}
    dirname_2=${SLURM_TMPDIR}/${partial_path_1}
    basename_1=ml_dataset_for_${ml_dataset_type}.h5
    basename_2=${basename_1}
    filename_1=${dirname_1}/${basename_1}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	cp ${filename_1} ${filename_2}
    fi
done

basename=execute_main_action_steps.py
script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${script_to_execute} \
       --ml_model_task=${ml_model_task} \
       --ml_model_idx=${ml_model_idx} \
       --data_dir_1=${SLURM_TMPDIR}
python_script_exit_code=$?

if [ "${python_script_exit_code}" -ne 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi

partial_path_3=ml_models/ml_model_${ml_model_idx}
dirname_1=${SLURM_TMPDIR}/${partial_path_3}
dirname_2=${path_to_data_dir_1}/${partial_path_3}

cd ${SLURM_TMPDIR}
mkdir -p ${dirname_2}
rm -rf ${dirname_2}
if [ "${dirname_1}" != "${dirname_2}" ]
then
    mv ${dirname_1} ${dirname_2}
fi

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    cd ${path_to_repo_root}
    rm -rf ${SLURM_TMPDIR}
fi
