#!/bin/bash
#SBATCH --job-name=generate_ml_dataset_for_ml_model_test_set_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=46G               # memory per node
#SBATCH --time=00-11:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL

path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
ml_model_task=${4}
disk_size_idx=${5}
disk_size=${6}
ml_dataset_idx=${7}
overwrite_slurm_tmpdir=${8}

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    SLURM_TMPDIR=${path_to_data_dir_1}
fi

basename=custom_env_setup_for_slurm_jobs.sh
if [ ! -f ${path_to_repo_root}/${basename} ]
then
    basename=default_env_setup_for_slurm_jobs.sh
fi
source ${path_to_repo_root}/${basename} ${SLURM_TMPDIR}/tempenv true

sample_name=MoS2_on_amorphous_C

partial_path_1=examples/modelling/cbed/simulations/${sample_name}/data
partial_path_2=${partial_path_1}/cbed_pattern_generator_output
partial_path_3=${partial_path_2}/patterns_with_${disk_size}_sized_disks

dirname_1=${path_to_repo_root}/${partial_path_3}
for filename_1 in ${dirname_1}/*
do
    basename_1=$(basename "${filename_1}")
    basename_2=${basename_1}
    dirname_2=${SLURM_TMPDIR}/${partial_path_3}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	cp ${filename_1} ${filename_2}
    fi
done

path_to_data_dir_2=${dirname_2}

basename=execute_main_action_steps.py
script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${script_to_execute} \
       --ml_model_task=${ml_model_task} \
       --disk_size_idx=${disk_size_idx} \
       --disk_size=${disk_size} \
       --ml_dataset_idx=${ml_dataset_idx} \
       --data_dir_1=${SLURM_TMPDIR} \
       --data_dir_2=${path_to_data_dir_2}
python_script_exit_code=$?

if [ "${python_script_exit_code}" -ne 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi

partial_path_4=ml_datasets/ml_datasets_for_ml_model_test_set_1
partial_path_5=${partial_path_4}/ml_datasets_with_cbed_patterns_of_
partial_path_6=${partial_path_5}${sample_name}
partial_path_7=${partial_path_6}/ml_datasets_with_${disk_size}_sized_disks

dirname_1=${SLURM_TMPDIR}/${partial_path_7}
dirname_2=${path_to_data_dir_1}/${partial_path_7}
basename_1=ml_dataset_${ml_dataset_idx}.h5
basename_2=${basename_1}
filename_1=${dirname_1}/${basename_1}
filename_2=${dirname_2}/${basename_2}

cd ${SLURM_TMPDIR}
mkdir -p ${dirname_2}
if [ "${filename_1}" != "${filename_2}" ]
then
    mv ${filename_1} ${filename_2}
fi

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    cd ${path_to_repo_root}
    rm -rf ${SLURM_TMPDIR}
fi
