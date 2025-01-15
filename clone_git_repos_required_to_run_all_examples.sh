#!/bin/bash

path_1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
path_to_dir_containing_current_script=${path_1}

mkdir -p ${path_1}/required_git_repos
cd ${path_1}/required_git_repos

git clone https://gitlab.com/mrfitzpa/emicroml.git

git clone https://gitlab.com/mrfitzpa/embeam.git
git clone https://gitlab.com/mrfitzpa/prismatique.git

cd -
