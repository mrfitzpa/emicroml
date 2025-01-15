#!/bin/bash
#SBATCH --job-name=generate_cbed_pattern_set
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32         # CPU cores/threads
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=0               # memory per node
#SBATCH --time=00-02:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL

path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
disk_size=${4}
overwrite_slurm_tmpdir=${5}

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

partial_path_1=potential_slice_generator_output

dirname_1=${path_to_data_dir_1}/${partial_path_1}
for filename_1 in ${dirname_1}/*
do
    basename_1=$(basename "${filename_1}")
    basename_2=${basename_1}
    dirname_2=${SLURM_TMPDIR}/${partial_path_1}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	cp ${filename_1} ${filename_2}
    fi
done

sample_name=MoS2

partial_path_2=examples/modelling/cbed/simulations
partial_path_3=${sample_name}/data
    
dirname_1=${path_to_repo_root}/${partial_path_2}/${partial_path_3}
dirname_2=${SLURM_TMPDIR}/${partial_path_3}
basename_1=sample_model_params_subset.json
basename_2=${basename_1}
filename_1=${dirname_1}/${basename_1}
filename_2=${dirname_2}/${basename_2}

mkdir -p ${dirname_2}
if [ "${filename_1}" != "${filename_2}" ]
then
    cp ${filename_1} ${filename_2}
fi

path_to_data_dir_2=${dirname_2}

basename=execute_main_action_steps.py
script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${script_to_execute} \
       --disk_size=${disk_size} \
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

partial_path_4=cbed_pattern_generator_output
partial_path_5=${partial_path_4}/patterns_with_${disk_size}_sized_disks

dirname_1=${SLURM_TMPDIR}/${partial_path_5}
dirname_2=${path_to_data_dir_1}/${partial_path_5}

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
