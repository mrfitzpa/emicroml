#!/bin/bash

current_machine_is_on_a_drac_server=false

dns_domain_name_of_current_machine=$(hostname -d)
if [ -z "${dns_domain_name_of_current_machine}" ]
then
    dns_domain_name_of_current_machine=$(hostname | grep -oP '(?<=\.).*$')
fi

readarray -t drac_dns_domain_names < ${path_to_repo_root}/drac_dns_domain_names

for drac_dns_domain_name in "${drac_dns_domain_names[@]}"
do
    if [ "${drac_dns_domain_name}" = "${dns_domain_name_of_current_machine}" ]
    then
	current_machine_is_on_a_drac_server=true
	break
    fi
done



cmd="realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)""
path_to_repo_root=$(${cmd})

path_to_temp_dir=${path_to_repo_root}/temp_${SLURM_JOB_ID}
mkdir -p ${path_to_temp_dir}



if [ "${current_machine_is_on_a_drac_server}" = true ]
then
    if [ $# -eq 0 ]
    then
	path_to_virtual_env=~/emicroml
	install_libs_required_to_run_all_examples=false
    else
	path_to_virtual_env=$1
	install_libs_required_to_run_all_examples=$2
    fi

    
    
    cmd="realpath "$(dirname "${path_to_virtual_env}")""
    path_to_parent_dir_of_virtual_env=$(${cmd})
    
    mkdir -p ${path_to_parent_dir_of_virtual_env}



    source ${path_to_repo_root}/load_drac_modules.sh


    
    virtualenv --no-download ${path_to_virtual_env}
    source ${path_to_virtual_env}/bin/activate
    pip install --no-index --upgrade pip

    

    pkgs="numpy numba hyperspy h5py pyFAI pytest ipympl jupyter"
    pkgs=${pkgs}" 'fakecbed>=0.3.0' h5pywrappers torch kornia"
    if [ "${install_libs_required_to_run_all_examples}" = true ]
    then
	pkgs=${pkgs}" pyprismatic-gpu"
    fi
    pip install --no-index ${pkgs}

    
    
else
    if [ $# -eq 0 ]
    then
	virtual_env_name=emicroml
	install_libs_required_to_run_all_examples=false
    else
	virtual_env_name=$1
	install_libs_required_to_run_all_examples=$2
    fi



    nvidia_smi_cmd_1="nvidia-smi"

    nvidia_smi_cmd_2="/c/Program\ Files/NVIDIA\ Corporation/NVSMI"
    nvidia_smi_cmd_2=${nvidia_smi_cmd_2}"/nvidia-smi.exe"

    nvidia_smi_cmd_3="/c/Windows/System32/DriverStore/FileRepository/nvdm*"
    nvidia_smi_cmd_3=${nvidia_smi_cmd_3}"/nvidia-smi.exe"

    declare -a nvidia_smi_cmds=("${nvidia_smi_cmd_1}"
				"${nvidia_smi_cmd_2}"
				"${nvidia_smi_cmd_3}")

    for nvidia_smi_cmd in "${nvidia_smi_cmds[@]}"
    do
	cmd_seq=${nvidia_smi_cmd}
	cmd_seq=${cmd_seq}" | grep -oP '(?<=CUDA Version: )'.*"
	cmd_seq=${cmd_seq}"| grep -oP '([1-9]+)' | head -1"
	major_cuda_version="$(eval "${cmd_seq}")"

	if [ "$?" -eq 0 ];
	then
	    break
	fi
    done



    if [ "${major_cuda_version}" = 11 ]
    then
	extra_torch_install_args="pytorch-cuda=11.8 -c pytorch -c nvidia"
	pyprismatic_pkg="pyprismatic=*=gpu*"
    elif [ "${major_cuda_version}" -gt 11 ]
    then
	extra_torch_install_args="pytorch-cuda=12.1 -c pytorch -c nvidia"
	pyprismatic_pkg="pyprismatic=*=gpu*"
    else
	extra_torch_install_args="cpuonly -c pytorch"
	pyprismatic_pkg="pyprismatic=*=cpu*"
    fi



    pkgs="python=3.10 numpy numba hyperspy h5py pyFAI pytest ipympl jupyter"
    pkgs=${pkgs}" 'fakecbed>=0.3.0' h5pywrappers"
    conda create -n ${virtual_env_name} ${pkgs} -y -c conda-forge
    conda activate ${virtual_env_name}



    conda install -y pytorch ${extra_torch_install_args}
    conda install -y kornia -c conda-forge



    if [ "${install_libs_required_to_run_all_examples}" = true ]
    then
	conda install -y ${pyprismatic_pkg} -c conda-forge
    fi
fi



path_to_required_git_repos=${path_to_repo_root}/required_git_repos
path_to_copy_of_required_git_repos=${path_to_temp_dir}/required_git_repos
cp -r ${path_to_required_git_repos} ${path_to_copy_of_required_git_repos}

libs=(emicroml)
if [ "${install_libs_required_to_run_all_examples}" = true ]
then
    libs+=(emconstants embeam prismatique)
fi

for lib in "${libs[@]}"
do
    cd ${path_to_temp_dir}/required_git_repos/${lib}
    pip install .
done



cd ${path_to_repo_root}
rm -rf ${path_to_temp_dir}
