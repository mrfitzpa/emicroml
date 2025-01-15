#!/bin/bash
#SBATCH --job-name=run_learning_rate_range_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=46G               # memory per node
#SBATCH --time=00-00:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL

path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
ml_model_task=${4}
test_idx=${5}
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

ml_dataset_types=(training)

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
       --test_idx=${test_idx} \
       --data_dir_1=${SLURM_TMPDIR}
python_script_exit_code=$?

if [ "${python_script_exit_code}" -ne 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi

if [ ${ml_model_task} = "cbed/fg/segmentation" ]
then
    architecture=resunet
elif [ ${ml_model_task} = "cbed/disk/segmentation" ]
then
    accepted_architectures=(deep_supervision_resunet)
    architecture=${accepted_architectures[${test_idx}]}
elif [ ${ml_model_task} = "cbed/distortion/estimation" ]
then
    architecture=multi_head_mod_resnet_57
else
    exit 1
fi

partial_path_3=learning_rate_range_test_results
partial_path_4=test_${test_idx}/using_${architecture}_architecture
partial_path_5=${partial_path_3}/${partial_path_4}

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
