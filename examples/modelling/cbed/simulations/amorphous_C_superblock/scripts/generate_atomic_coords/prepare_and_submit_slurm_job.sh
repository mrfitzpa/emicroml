#!/bin/bash
#SBATCH --job-name=generate_atomic_coords
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=4G               # memory per node
#SBATCH --time=00-02:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL

path_to_dir_containing_current_script=${1}
path_to_repo_root=${2}
path_to_data_dir_1=${3}
overwrite_slurm_tmpdir=${4}

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

sample_names=(amorphous_C_block MoS2)

for sample_name in "${sample_names[@]}"
do
    partial_path_1=examples/modelling/cbed/simulations/${sample_name}/data
    
    dirname_1=${path_to_repo_root}/${partial_path_1}
    dirname_2=${SLURM_TMPDIR}/${sample_name}/data
    basename_1=atomic_coords.xyz
    basename_2=${basename_1}
    filename_1=${dirname_1}/${basename_1}
    filename_2=${dirname_2}/${basename_2}

    mkdir -p ${dirname_2}
    if [ "${filename_1}" != "${filename_2}" ]
    then
	cp ${filename_1} ${filename_2}
    fi
done

path_to_data_dir_2=${SLURM_TMPDIR}/amorphous_C_block/data
path_to_data_dir_3=${SLURM_TMPDIR}/MoS2/data

basename=execute_main_action_steps.py
script_to_execute=${path_to_dir_containing_current_script}/${basename}

python ${script_to_execute} \
       --data_dir_1=${SLURM_TMPDIR} \
       --data_dir_2=${path_to_data_dir_2} \
       --data_dir_3=${path_to_data_dir_3}
python_script_exit_code=$?

if [ "${python_script_exit_code}" -ne 0 ];
then
    msg="\n\n\nThe slurm job terminated early with at least one error. "
    msg=${msg}"See traceback for details.\n\n\n"
    echo -e ${msg}
    exit 1
fi

dirname_1=${SLURM_TMPDIR}
dirname_2=${path_to_data_dir_1}
basename_1=atomic_coords.xyz
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
