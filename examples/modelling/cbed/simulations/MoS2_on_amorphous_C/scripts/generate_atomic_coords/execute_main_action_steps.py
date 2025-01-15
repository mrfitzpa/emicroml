"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For making directories and creating path objects.
import pathlib



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



# Read in the atomic coordinates of the MoS2 sub-sample.
filename = path_to_data_dir_2 + "/atomic_coords.xyz"
with open(filename, "r") as file_obj:
    throw_away_line = file_obj.readline()

    line = file_obj.readline()
    Delta_X = float(line.split()[0])
    Delta_Y = float(line.split()[1])
    Delta_Z_of_MoS2 = float(line.split()[2])

    last_line_has_not_been_read = True

    Mo_sample_unit_cell = []
    S_sample_unit_cell = []

    atomic_potential_extent = np.inf

    while last_line_has_not_been_read:
        line = file_obj.readline()
        numbers_from_lines = tuple(float(string) for string in line.split())
        if len(numbers_from_lines) == 1:
            last_line_has_not_been_read = False
        else:
            Z, x, y, z_prime, occ, u_x_rms = numbers_from_lines
            atomic_potential_extent = min(z_prime, atomic_potential_extent)
            if Z == 42:
                Mo_sample_unit_cell.append([x, y, z_prime])
            else:
                S_sample_unit_cell.append([x, y, z_prime])

Mo_sample_unit_cell = np.array(Mo_sample_unit_cell)
S_sample_unit_cell = np.array(S_sample_unit_cell)



# Read in the atomic coordinates of the amorphous C superblock and shift them
# such that the amorphous C superblock lies above the MoS2 sub-sample, i.e. the
# electron beam makes contact with the amorphous C superblock first.
filename = path_to_data_dir_3 + "/atomic_coords.xyz"
with open(filename, "r") as file_obj:
    throw_away_line = file_obj.readline()
    throw_away_line = file_obj.readline()

    last_line_has_not_been_read = True

    C_sample_unit_cell = []

    z_prime_shift = Delta_Z_of_MoS2 - (atomic_potential_extent/2)
    Delta_Z = 0

    while last_line_has_not_been_read:
        line = file_obj.readline()
        numbers_from_lines = tuple(float(string) for string in line.split())
        if len(numbers_from_lines) == 1:
            last_line_has_not_been_read = False
        else:
            Z, x, y, z_prime, occ, u_x_rms = numbers_from_lines
            Delta_Z = max(z_prime+z_prime_shift+atomic_potential_extent,
                          Delta_Z)
            C_sample_unit_cell.append([x, y, z_prime+z_prime_shift])

C_sample_unit_cell = np.array(C_sample_unit_cell)



# Set the sample unit cell dimensions.
sample_unit_cell_dims = (Delta_X, Delta_Y, Delta_Z)



# Write atomic coordinates of target sample to file.
output_dirname = str(path_to_data_dir_1)
pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)

filename = output_dirname + "/atomic_coords.xyz"
with open(filename, "w") as file_obj:
    line = "MoS2 on Amorphous C Sample\n"
    file_obj.write(line)

    unformatted_line = "\t{:18.14f}\t{:18.14f}\t{:18.14f}\n"
    formatted_line = unformatted_line.format(*sample_unit_cell_dims)
    file_obj.write(formatted_line)

    occ = 1  # Number not used except to write file with expected format.

    Z_of_Mo = 42  # Atomic number of Mo.
    Z_of_S = 16  # Atomic number of S.
    Z_of_C = 6  # Atomic number of C.

    # The RMS x-displacement of Mo atoms at room temperature. Value was taken
    # from experimental data for the RMS of the in-plane displacement in
    # Schönfeld et al., Acta Cryst. B39, 404-407 (1983).
    u_x_rms_of_Mo = 0.069

    # The RMS x-displacement of S atoms at room temperature. Value was taken
    # from experimental data for the RMS of the in-plane displacement in
    # Schönfeld et al., Acta Cryst. B39, 404-407 (1983).
    u_x_rms_of_S = 0.062

    # The RMS x-displacement of C atoms at room temperature. Value was taken
    # from C. B. Boothroyd, Ultramicroscopy **83**, 3-4 (2000).
    u_x_rms_of_C = 0.141

    single_species_sample_unit_cells = (Mo_sample_unit_cell,
                                        S_sample_unit_cell,
                                        C_sample_unit_cell)
    Z_set = (Z_of_Mo, Z_of_S, Z_of_C)
    u_x_rms_set = (u_x_rms_of_Mo, u_x_rms_of_S, u_x_rms_of_C)
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
