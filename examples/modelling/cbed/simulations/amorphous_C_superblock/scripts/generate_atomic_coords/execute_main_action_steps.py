"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For making directories and creating path objects.
import pathlib

# For generating permutations of a sequence.
import itertools



# For general array handling.
import numpy as np



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2023"
__credits__      = ["Matthew Fitzpatrick"]
__maintainer__   = "Matthew Fitzpatrick"
__email__        = "mrfitzpa@uvic.ca"
__status__       = "Development"



###############################################
## Define classes, functions, and contstants ##
###############################################



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

# Parse command line arguments.
parser = argparse.ArgumentParser()
argument_names = ("data_dir_1", "data_dir_2", "data_dir_3")
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
path_to_data_dir_1 = args.data_dir_1
path_to_data_dir_2 = args.data_dir_2
path_to_data_dir_3 = args.data_dir_3



# The idea is to read-in the atomic coordinates of a 5x5x1 nm^3 large amorphous
# C block, and store this block for later use in some container. Next, we take
# the block in this container, transform the block in a variety of ways using
# reflections and axes permutations, to generate 25 5x5x1 nm^3 blocks, which we
# can then tile to construct our 25x25x1 nm^3 sized superblock, which we then
# truncate to the same lateral dimensions as the MoS2 sub-sample, and to 0.5 nm
# along the z-axis.



# Read in the atomic coordinates of the 5x5x1 nm^3 large amorphous C block.
filename = path_to_data_dir_2 + "/atomic_coords.xyz"
with open(filename, "r") as file_obj:
    throw_away_line = file_obj.readline()
        
    line = file_obj.readline()
    amorphous_C_block_tile_dims = tuple(float(s) for s in line.split())

    last_line_has_not_been_read = True

    amorphous_C_block_tile = []

    while last_line_has_not_been_read:
        line = file_obj.readline()
        numbers_from_lines = tuple(float(string) for string in line.split())
        if len(numbers_from_lines) == 1:
            last_line_has_not_been_read = False
        else:
            Z, x, y, z_prime, occ, u_x_rms = numbers_from_lines
            position_of_atom = (x, y, z_prime)
            amorphous_C_block_tile.append(position_of_atom)



# Generate the set of transformations to apply to blocks.
reflection_seq_set = []
for reflect_across_yz_plane in (False, True):
    for reflect_across_xz_plane in (False, True):
        for reflect_across_xy_plane in (False, True):
            reflection_seq = ((-1)**reflect_across_yz_plane,
                              (-1)**reflect_across_xz_plane,
                              (-1)**reflect_across_xy_plane)
            reflection_seq_set.append(reflection_seq)
                
axes_permutations = tuple(itertools.permutations([0, 1, 2]))



# Transform blocks to yield 25 blocks; and tile the 25 blocks to generate a
# 25x25x1 nm^3 superblock, which we will subsequently block.
block_tile_idx = 0
target_num_blocks = 25
amorphous_C_superblock_dims_in_blocks = (5, 5, 1)
amorphous_C_block_side_length = amorphous_C_block_tile_dims[0]
amorphous_C_superblock_sample_unit_cell = []

for reflection_seq in reflection_seq_set:
    for axes_permutation in axes_permutations:
        shift = np.unravel_index(block_tile_idx,
                                 amorphous_C_superblock_dims_in_blocks)
        shift = amorphous_C_block_side_length * np.array(shift)
        transformation_matrix = reflection_seq * np.eye(3)[axes_permutation, :]
        for position_of_seed_atom in amorphous_C_block_tile:
            position_of_atom = position_of_seed_atom @ transformation_matrix
            position_of_atom %= amorphous_C_block_tile_dims
            position_of_atom += shift
            amorphous_C_superblock_sample_unit_cell.append(position_of_atom)
        block_tile_idx += 1
        if block_tile_idx == target_num_blocks:
            break
    if block_tile_idx == target_num_blocks:
        break



# Read in the dimensions of the MoS2 sub-sample.
filename = path_to_data_dir_3 + "/atomic_coords.xyz"
with open(filename, "r") as file_obj:
    throw_away_line = file_obj.readline()
        
    line = file_obj.readline()
MoS2_subsample_dims = tuple(float(s) for s in line.split())
Delta_X, Delta_Y, _ = MoS2_subsample_dims  # In Å.



# Set the amorphous C superblock unit cell dimensions.
Delta_Z = 5  # In Å.
sample_unit_cell_dims = (Delta_X, Delta_Y, Delta_Z)



# Truncate the amorphous C superblock.
num_atoms_in_amorphous_C_superblock_pre_truncation = \
    len(amorphous_C_superblock_sample_unit_cell)

indices = range(num_atoms_in_amorphous_C_superblock_pre_truncation-1, -1, -1)
for idx in indices:
    position_of_atom = amorphous_C_superblock_sample_unit_cell[idx]
    x, y, z_prime = position_of_atom
    if (((x < 0) or (Delta_X <= x))
        or ((y < 0) or (Delta_Y <= y))
        or ((z_prime < 0) or (Delta_Z <= z_prime))):
        del amorphous_C_superblock_sample_unit_cell[idx]



# Write atomic coordinates of target sample to file.
output_dirname = str(path_to_data_dir_1)
pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)

filename = output_dirname + "/atomic_coords.xyz"
with open(filename, "w") as file_obj:
    line = "Amorphous C Superblock\n"
    file_obj.write(line)

    unformatted_line = "\t{:18.14f}\t{:18.14f}\t{:18.14f}\n"
    formatted_line = unformatted_line.format(*sample_unit_cell_dims)
    file_obj.write(formatted_line)

    occ = 1  # Number not used except to write file with expected format.

    Z_of_C = 6  # Atomic number of C.

    # The RMS x-displacement of C atoms at room temperature. Value was taken
    # from C. B. Boothroyd, Ultramicroscopy **83**, 3-4 (2000).
    u_x_rms_of_C = 0.141

    C_sample_unit_cell = amorphous_C_superblock_sample_unit_cell
    single_species_sample_unit_cells = (C_sample_unit_cell,)
    Z_set = (Z_of_C,)
    u_x_rms_set = (u_x_rms_of_C,)
    zip_obj = zip(single_species_sample_unit_cells, Z_set, u_x_rms_set)
        
    for triplet in zip_obj:
        single_species_sample_unit_cell, Z, u_x_rms = triplet
        for position_of_atom in single_species_sample_unit_cell:
            x, y, z_prime = position_of_atom
            unformatted_line = ("{}\t{:18.14f}\t{:18.14f}"
                                "\t{:18.14f}\t{:18.14f}\t{:18.14f}\n")
            args = (Z, x, y, z_prime, occ, u_x_rms)
            formatted_line = unformatted_line.format(*args)
            file_obj.write(formatted_line)

    file_obj.write("-1")
