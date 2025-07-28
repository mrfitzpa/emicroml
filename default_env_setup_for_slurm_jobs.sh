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



# The current script will attempt to create a virtual environment, then activate
# it within the current shell, and then install the ``emicroml`` library within
# said environment, along with some optional libraries. If the script is
# executed on a Digital Alliance of Canada (DRAC) high-performance computing
# (HPC) server, then the virtual environment is created via
# ``virtualenv``. Otherwise, the virtual environment is created via
# ``conda``. For the latter scenario, an ``anaconda`` or ``miniconda``
# distribution must be installed prior to running the script.
#
# The correct form of the command to run the script is::
#
#  source <path_to_current_script> <env_name> <install_extras>
#
# where ``<path_to_current_script>`` is the absolute or relative path to the
# current script; ``<env_name>`` is the path to the virtual environment, if the
# script is being executed on a DRAC HPC server, else it is the name of the
# ``conda`` virtual environment; and ``<install_extras>`` is a boolean, i.e. it
# should either be ``true`` or ``false``. If ``<install_extras>`` is set to
# ``true``, then the script will attempt to install within the environment the
# dependencies required to run all of the examples in the repository, in
# addition to installing ``emicroml``. Otherwise, the script will attempt to
# install only ``emicroml`` and its dependencies, i.e. not the additional
# libraries required to run the examples.
#
# If the virtual environment is to be created on either the ``narval``,
# ``beluga``, or ``graham`` HPC server belonging to DRAC, and the script with
# the basename ``download_wheels_for_offline_env_setup_on_drac_server.sh`` at
# the root of the repository has never been executed, then one must first change
# into the root of the repository, and subsequently execute that script via the
# following command::
#
#  bash download_wheels_for_offline_env_setup_on_drac_server.sh
#
# Upon completion of that script, a set of Python wheels will be downloaded to
# the directory ``<root>/_wheels_for_offline_env_setup_on_drac_server``, where
# ``<root>`` is the root of the repository. Note that that script only needs to
# be executed once, assuming one does not modify or delete the directory
# ``<root>/_wheels_for_offline_env_setup_on_drac_server``.



# Begin timer, and print starting message.
start_time=$(date +%s.%N)

msg="Beginning virtual environment creation and setup..."
echo ""
echo ${msg}
echo ""
echo ""
echo ""



# Get the path to the root of the repository.
cmd="realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)""
path_to_repo_root=$(${cmd})



# Automatically determine whether the script is being executed on a DRAC HPC
# server. Also determine whether we need to install some libraries from wheels
# that were previously downloaded.
current_machine_is_on_a_drac_server=false
drac_compute_nodes_have_internet_access=false

dns_domain_name_of_current_machine=$(hostname -d)
if [ -z "${dns_domain_name_of_current_machine}" ]
then
    dns_domain_name_of_current_machine=$(hostname | grep -oP '(?<=\.).*$')
fi

basename=drac_dns_domain_name_to_internet_accessibility_map
path_to_file_to_read=${path_to_repo_root}/${basename}

while read line
do
    key_val_pair=(${line})
    key=${key_val_pair[0]}
    val=${key_val_pair[1]}

    drac_dns_domain_name=${key}
    if [ "${key}" = "${dns_domain_name_of_current_machine}" ]
    then
    	current_machine_is_on_a_drac_server=true

    	internet_accessibility=${val}
    	if [ "${val}" = "compute_nodes_have_internet_access" ]
    	then
    	    drac_compute_nodes_have_internet_access=true
    	fi	

    	break
    fi
done < ${path_to_file_to_read}



# Create a temporary directory, in which to install ``emicroml`` further below.
path_to_temp_dir=${path_to_repo_root}/temp_${SLURM_JOB_ID}
mkdir -p ${path_to_temp_dir}



if [ "${current_machine_is_on_a_drac_server}" = true ]
then
    # Parse the command line arguments.
    if [ $# -eq 0 ]
    then
	path_to_virtual_env=~/emicroml
	install_libs_required_to_run_all_examples=false
    else
	path_to_virtual_env=$1
	install_libs_required_to_run_all_examples=$2
    fi



    # Load some DRAC software modules.
    module load StdEnv/2023
    module load python/3.11 hdf5 cuda



    # Create the virtual environment, activate it, and then upgrade ``pip``.
    cmd="realpath "$(dirname "${path_to_virtual_env}")""
    path_to_parent_dir_of_virtual_env=$(${cmd})
    
    mkdir -p ${path_to_parent_dir_of_virtual_env}
    
    virtualenv --no-download ${path_to_virtual_env}
    source ${path_to_virtual_env}/bin/activate
    pip install --no-index --upgrade pip

    

    # Install the remaining libraries in the virtual environment, except for
    # ``emicroml``. Where applicable, GPU-supported versions of libraries are
    # installed.
    pkgs="numpy<2.0.0 numba hyperspy h5py pytest ipympl jupyter torch kornia"
    pkgs=${pkgs}" blosc2 msgpack"
    if [ "${install_libs_required_to_run_all_examples}" = true ]
    then
	pkgs=${pkgs}" pyopencl pyFAI pyprismatic-gpu"
    fi
    pip install --no-index ${pkgs}

    if [ "${drac_compute_nodes_have_internet_access}" = false ]
    then
	cd ${path_to_repo_root}/_wheels_for_offline_env_setup_on_drac_server

	pkgs="czekitout*.whl fancytypes*.whl h5pywrappers*.whl"
	pkgs=${pkgs}" distoptica*.whl fakecbed*.whl"
	if [ "${install_libs_required_to_run_all_examples}" = true ]
	then
	    pkgs=${pkgs}" empix*.whl embeam*.whl prismatique*.whl"
	fi
	pip install ${pkgs}
    else
	pkgs="fakecbed>=0.3.6 h5pywrappers"
	if [ "${install_libs_required_to_run_all_examples}" = true ]
	then
	    pkgs=${pkgs}" prismatique"
	fi
	pip install ${pkgs}
    fi	

    if [ "${install_libs_required_to_run_all_examples}" = true ]
    then
	pkgs="pyopencl pyFAI pyprismatic-gpu"
	pip install --no-index ${pkgs}

	pkgs="prismatique"
	pip install ${pkgs}
    fi
else
    # Parse the command line arguments.
    if [ $# -eq 0 ]
    then
	virtual_env_name=emicroml
	install_libs_required_to_run_all_examples=false
    else
	virtual_env_name=$1
	install_libs_required_to_run_all_examples=$2
    fi



    # Determine automatically whether NVIDIA drivers have been installed. If
    # they have been installed, then the script will install GPU-supported
    # versions of certain libraries.
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
	${nvidia_smi_cmd} 2>/dev/null
	if [ "$?" -ne 0 ]
	then
	    major_cuda_version="0"
	    continue
	fi

	cmd_seq=${nvidia_smi_cmd}
	cmd_seq=${cmd_seq}" | grep -oP '(?<=CUDA Version: )'.*"
	cmd_seq=${cmd_seq}"| grep -oP '([1-9]+)' | head -1"
	major_cuda_version="$(eval "${cmd_seq}")"

	if [ "$?" -eq 0 ]
	then
	    break
	fi
    done



    # Determine which versions of ``pytorch`` and ``pyprismatic`` to installs
    # according to what NVIDIA drivers are installed, if any.
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



    # Create the ``conda`` virtual environment and install a subset of
    # libraries, then activate the virtual environment.
    pkgs="python=3.11 numpy numba hyperspy h5py pytest ipympl jupyter"
    pkgs=${pkgs}" fakecbed>=0.3.6 h5pywrappers"
    conda create -n ${virtual_env_name} ${pkgs} -y -c conda-forge
    conda activate ${virtual_env_name}



    # Install the remaining libraries in the virtual environment, except
    # for``emicroml``.
    conda install -y pytorch ${extra_torch_install_args}
    conda install -y kornia -c conda-forge

    if [ "${install_libs_required_to_run_all_examples}" = true ]
    then
	pkgs="pyopencl[pocl] pyFAI"
	pip install ${pkgs}
	
	conda install -y ${pyprismatic_pkg} -c conda-forge

	pkgs="prismatique"
	pip install ${pkgs}
    fi
fi



# Install ``emicroml`` and then remove the temporary directory that we created
# above.
path_to_required_git_repos=${path_to_repo_root}/required_git_repos
path_to_copy_of_required_git_repos=${path_to_temp_dir}/required_git_repos
cp -r ${path_to_required_git_repos} ${path_to_copy_of_required_git_repos}

cd ${path_to_copy_of_required_git_repos}/emicroml
git status
pip install .

cd ${path_to_repo_root}
rm -rf ${path_to_temp_dir}



# End timer and print completion message.
end_time=$(date +%s.%N)
elapsed_time=$(echo "${end_time} - ${start_time}" | bc -l)

echo ""
echo ""
echo ""
msg="Finished virtual environment creation and setup. Time taken: "
msg=${msg}"${elapsed_time} s."
echo ${msg}
echo ""
