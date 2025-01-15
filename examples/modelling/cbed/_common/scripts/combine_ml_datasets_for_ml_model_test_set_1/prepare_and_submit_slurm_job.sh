#!/bin/bash
#SBATCH --job-name=combine_ml_datasets_for_model_test_set_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=4G               # memory per node
#SBATCH --time=00-02:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL

path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
ml_model_task=${4}
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
source ${path_to_repo_root}/${basename} ${SLURM_TMPDIR}/tempenv false

cd ${path_to_data_dir_1}

sample_name=MoS2_on_amorphous_C

partial_path_1=ml_datasets/ml_datasets_for_ml_model_test_set_1
partial_path_2=${partial_path_1}/ml_datasets_with_cbed_patterns_of_
partial_path_3=${partial_path_2}${sample_name}

for partial_path_4 in ${partial_path_3}/ml_datasets_with_*_sized_disks
do
    dirname_1=${path_to_data_dir_1}/${partial_path_4}
    for filename_1 in ${dirname_1}/ml_dataset_*.h5
    do
	basename_1=$(basename "${filename_1}")
	basename_2=${basename_1}
	dirname_2=${SLURM_TMPDIR}/${partial_path_4}
	filename_2=${dirname_2}/${basename_2}

	mkdir -p ${dirname_2}
	if [ "${filename_1}" != "${filename_2}" ]
	then
	    cp ${filename_1} ${filename_2}
	fi
    done
done

cd ${SLURM_TMPDIR}

basename=execute_main_action_steps.py
script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${script_to_execute} \
       --ml_model_task=${ml_model_task} \
       --data_dir_1=${SLURM_TMPDIR}
python_script_exit_code=$?

if [ "${python_script_exit_code}" -ne 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi

cd ${SLURM_TMPDIR}

for partial_path_5 in ${partial_path_3}/ml_dataset_with_*_sized_disks.h5
do
    dirname_1=${SLURM_TMPDIR}/${partial_path_3}
    filename_1=${SLURM_TMPDIR}/${partial_path_5}

    basename_1=$(basename "${filename_1}")
    basename_2=${basename_1}
    
    dirname_2=${path_to_data_dir_1}/${partial_path_3}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	mv ${filename_1} ${filename_2}
    fi
done

cd ${path_to_data_dir_1}

for partial_path_4 in ${partial_path_3}/ml_datasets_with_*_sized_disks
do
    rm -rf ${path_to_data_dir_1}/${partial_path_4}
done

if [ "${overwrite_slurm_tmpdir}" = true ]
then
    cd ${path_to_repo_root}
    rm -rf ${SLURM_TMPDIR}
fi
