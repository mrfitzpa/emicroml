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
# current script. The parent script performs the "action" of running a single
# machine learning (ML) model test from the "first" test set, for a specified
# task.
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
#   4. Moves non-temporary output data that is generated from the main steps to
#   their expected final destinations.
#
#   5. Deletes/removes any remaining temporary files or directories.
#
# The main steps are executed by running the script with the basename
# ``execute_main_action_steps.py``, located in the same directory as the current
# script.



#SBATCH --job-name=run_ml_model_test_set_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6               # CPU cores/threads
#SBATCH --gpus-per-node=a100_3g.20gb:1  # GPU type and number of GPUs per node.
#SBATCH --mem=62G                       # CPU memory per node
#SBATCH --time=00-01:59                 # time (DD-HH:MM)
#SBATCH --mail-type=ALL



# Parse the command line arguments.
path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
ml_model_task=${4}
overwrite_slurm_tmpdir=${5}



# Overwrite ``SLURM_TMPDIR`` if specified to do so.
if [ "${overwrite_slurm_tmpdir}" = true ]
then
    SLURM_TMPDIR=${path_to_data_dir_1}/tmp_dir
fi



# Setup the Python virtual environment used to execute the main action steps.
basename=custom_env_setup_for_slurm_jobs.sh
if [ ! -f ${path_to_repo_root}/${basename} ]
then
    basename=default_env_setup_for_slurm_jobs.sh
fi
source ${path_to_repo_root}/${basename} ${SLURM_TMPDIR}/tempenv false



# Copy the input data to the temporary directories.
cd ${path_to_data_dir_1}

partial_path_1=ml_models

for partial_path_2 in ${partial_path_1}/ml_model_*
do
    dirname_1=${path_to_data_dir_1}/${partial_path_2}
    for filename_1 in ${dirname_1}/ml_model_at_lr_step_*.pth
    do
	basename_1=$(basename "${filename_1}")
	basename_2=${basename_1}
	dirname_2=${SLURM_TMPDIR}/${partial_path_2}
	filename_2=${dirname_2}/${basename_2}

	mkdir -p ${dirname_2}
	if [ "${filename_1}" != "${filename_2}" ]
	then
	    cp ${filename_1} ${filename_2}
	    msg="Copied file at ``'"${filename_1}"'`` to ``'"${filename_2}"'``."
	    echo ${msg}
	    echo ""
	fi
    done
done

sample_name=MoS2_on_amorphous_C

partial_path_3=ml_datasets/ml_datasets_for_ml_model_test_set_1
partial_path_4=${partial_path_3}/ml_datasets_with_cbed_patterns_of_
partial_path_5=${partial_path_4}${sample_name}

dirname_1=${path_to_data_dir_1}/${partial_path_5}
for filename_1 in ${dirname_1}/ml_dataset_with_*_sized_disks.h5
do
    basename_1=$(basename "${filename_1}")
    basename_2=${basename_1}
    dirname_2=${SLURM_TMPDIR}/${partial_path_5}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	cp ${filename_1} ${filename_2}
	msg="Copied file at ``'"${filename_1}"'`` to ``'"${filename_2}"'``."
	echo ${msg}
	echo ""
    fi
done

echo ""
echo ""

cd ${SLURM_TMPDIR}



# Execute the script that executes the main action steps.
basename=execute_main_action_steps.py
path_to_script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${path_to_script_to_execute} \
       --ml_model_task=${ml_model_task} \
       --data_dir_1=${SLURM_TMPDIR}
python_script_exit_code=$?

if [ "${python_script_exit_code}" != 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi



# Move the non-temporary output data that is generated from the main steps to
# their expected final destinations. Also delete/remove any remaining temporary
# files or directories.
cd ${SLURM_TMPDIR}

for partial_path_2 in ${partial_path_1}/ml_model_*
do
    partial_path_6=${partial_path_2}/ml_model_test_set_1_results
    for partial_path_7 in ${partial_path_6}/results_for_cbed_patterns_of_*
    do
	dirname_1=${SLURM_TMPDIR}/${partial_path_7}
	filename_1=${dirname_1}/ml_model_testing_summary_output_data.h5

	basename_1=$(basename "${filename_1}")
	basename_2=${basename_1}
    
	dirname_2=${path_to_data_dir_1}/${partial_path_7}
	filename_2=${dirname_2}/${basename_2}

	mkdir -p ${dirname_2}
	if [ "${filename_1}" != "${filename_2}" ]
	then
	    mv ${filename_1} ${filename_2}
	    msg="Moved file at ``'"${filename_1}"'`` to ``'"${filename_2}"'``."
	    echo ${msg}
	    echo ""
	fi
    done
done

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    cd ${path_to_repo_root}
    rm -rf ${SLURM_TMPDIR}
fi
