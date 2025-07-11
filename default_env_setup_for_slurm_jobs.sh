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
# server.
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
    source ${path_to_repo_root}/load_drac_modules.sh



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
    pkgs="numpy numba hyperspy h5py pytest ipympl jupyter torch kornia"
    pkgs=${pkgs}" blosc2 msgpack"
    pip install --no-index ${pkgs}

    pkgs="fakecbed>=0.3.6 h5pywrappers"
    pip install ${pkgs}

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
    pkgs="python=3.10 numpy numba hyperspy h5py pytest ipympl jupyter"
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



# Install ``emicroml``.
cd ${path_to_repo_root}
pip install .



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
