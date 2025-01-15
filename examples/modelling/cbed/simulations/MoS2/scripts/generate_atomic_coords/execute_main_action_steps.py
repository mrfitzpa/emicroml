"""Insert here a brief description of the package.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For parsing command line arguments.
import argparse

# For making directories and creating path objects.
import pathlib

# For serializing JSON objects.
import json



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
argument_names = ("data_dir_1",)
for argument_name in argument_names:
    parser.add_argument("--"+argument_name)
args = parser.parse_args()
path_to_data_dir_1 = args.data_dir_1



# "a" and "c" lattice parameters of MoS2 in Å.
a = 3.1604
c = 12.295

# MoS2 unit-cell lattice vectors. These vectors will yield a hexagonal lattice.
a_1 = a * np.array([1.0, 0.0, 0.0])
a_2 = a * np.array([0.0, np.sqrt(3), 0.0])
a_3 = c * np.array([0.0, 0.0, 1.0])

# Used to define the positions of atoms in unit cell.
u = 0.612

# Positions of S atoms in unit cell.
delta_S_1 = (1/2)*a_1 + (1/6)*a_2 + (u-1/2)*a_3
delta_S_2 = (1/2)*a_1 + (1/6)*a_2 + (1-u)*a_3
delta_S_3 = (2/3)*a_2 + (u-1/2)*a_3
delta_S_4 = (2/3)*a_2 + (1-u)*a_3
delta_S_5 = (1/3)*a_2 + u*a_3
delta_S_6 = (1/3)*a_2 + (3/2-u)*a_3
delta_S_7 = (1/2)*a_1 + (5/6)*a_2 + u*a_3
delta_S_8 = (1/2)*a_1 + (5/6)*a_2 + (3/2-u)*a_3

# Positions of Mo atoms in unit cell.
delta_Mo_1 = (1/3)*a_2 + (1/4)*a_3
delta_Mo_2 = (1/2)*a_1 + (5/6)*a_2 + (1/4)*a_3
delta_Mo_3 = (1/2)*a_1 + (1/6)*a_2 + (3/4)*a_3
delta_Mo_4 = (2/3)*a_2 + (3/4)*a_3

# Define the orthorhombic MoS2 unit cell in two parts.
S_unit_cell = np.array([delta_S_1, delta_S_2, delta_S_3, delta_S_4,
                        delta_S_5, delta_S_6, delta_S_7, delta_S_8])
Mo_unit_cell = np.array([delta_Mo_1, delta_Mo_2, delta_Mo_3, delta_Mo_4])

# Magnitude of unit-cell lattice vectors.
a_1_mag = np.linalg.norm(a_1)
a_2_mag = np.linalg.norm(a_2)

# Magnitude of either primitive reciprocal lattice vector.
b_1_mag = (2*np.pi) * (2/a/np.sqrt(3))

# Subset of discretization parameters of real-space.
sample_supercell_reduced_xy_dims_in_pixels = (1024, 1024)
interpolation_factors = (1, 1)



# Determine the number of tiles of the MoS2 unit cell to form the sample unit
# cell.
tilde_N_x = sample_supercell_reduced_xy_dims_in_pixels[0]
f_x = interpolation_factors[0]
N_x = 4*f_x*tilde_N_x
upper_limit_of_Delta_X = np.pi*N_x/(24*b_1_mag)
num_atomic_layers = 5

num_x_tiles = int(np.ceil(upper_limit_of_Delta_X / a_1_mag))
Delta_X = a_1_mag*num_x_tiles
ratio = 0
while (ratio < 0.99) or (Delta_X > upper_limit_of_Delta_X):
    num_x_tiles -= 1
    Delta_X = a_1_mag*num_x_tiles
    num_y_tiles = int(np.round(Delta_X / a_2_mag))
    Delta_Y = a_2_mag*num_y_tiles
    numerator = min(Delta_X, Delta_Y)
    denominator = max(Delta_X, Delta_Y)
    ratio = numerator/denominator



# We tile the MoS2 unit cell to form the sample unit cell.
x_tiling_indices = range(-(num_x_tiles//2), -(num_x_tiles//2)+num_x_tiles)
y_tiling_indices = range(-(num_y_tiles//2), -(num_y_tiles//2)+num_y_tiles)
z_tiling_indices = range(0, (num_atomic_layers//2)+(num_atomic_layers%2))

Mo_sample_unit_cell = []
S_sample_unit_cell = []

for x_tiling_idx in x_tiling_indices:
    for y_tiling_idx in y_tiling_indices:
        for z_tiling_idx in z_tiling_indices:
            shift = x_tiling_idx*a_1 + y_tiling_idx*a_2 + z_tiling_idx*a_3

            if ((z_tiling_idx == max(z_tiling_indices))
                and (num_atomic_layers%2 == 1)):
                current_Mo_cell = np.array([delta_Mo+shift
                                            for delta_Mo
                                            in Mo_unit_cell[:2]])
                current_S_cell = np.array([delta_S+shift
                                           for delta_S
                                           in S_unit_cell[:4]])
            else:
                current_Mo_cell = np.array([delta_Mo+shift
                                            for delta_Mo
                                            in Mo_unit_cell])
                current_S_cell = np.array([delta_S+shift
                                           for delta_S
                                           in S_unit_cell])

            for position_of_current_atom in current_Mo_cell:
                x, y, z_prime = position_of_current_atom
                Mo_sample_unit_cell.append((x, y, z_prime))

            for position_of_current_atom in current_S_cell:
                x, y, z_prime = position_of_current_atom
                S_sample_unit_cell.append((x, y, z_prime))

Mo_sample_unit_cell = np.array(Mo_sample_unit_cell)
S_sample_unit_cell = np.array(S_sample_unit_cell)



# Find minimum and maximum x-, y-, and z-coordinates of the sample.
single_species_sample_unit_cells = (Mo_sample_unit_cell, S_sample_unit_cell)

min_coords = [np.inf, np.inf, np.inf]
max_coords = [-np.inf, -np.inf, -np.inf]

for axis in range(3):
    sample_unit_cells = single_species_sample_unit_cells
    for single_species_sample_unit_cell in sample_unit_cells:
        candidate_min_coord = np.amin(single_species_sample_unit_cell[:, axis])
        candidate_max_coord = np.amax(single_species_sample_unit_cell[:, axis])
        min_coords[axis] = min(candidate_min_coord, min_coords[axis])
        max_coords[axis] = max(candidate_max_coord, max_coords[axis])



# Determine sample supercell dimensions, keeping in mind we want a buffer equal
# to the atomic potential extent from each edge of the sample supercell.
atomic_potential_extent = 3  # In Å.
Delta_Z = (max_coords[2]-min_coords[2]) + 2*atomic_potential_extent
sample_unit_cell_dims = (Delta_X, Delta_Y, Delta_Z)



# Apply a global shift to the atomic coordinates so that they are all positive.
global_shift = np.array([min_coords[0],
                         min_coords[1],
                         min_coords[2]-atomic_potential_extent])

for idx, _ in enumerate(Mo_sample_unit_cell):
    Mo_sample_unit_cell[idx] -= global_shift
for idx, _ in enumerate(S_sample_unit_cell):
    S_sample_unit_cell[idx] -= global_shift



# Write to file some of the parameters used to construct the model of the MoS2
# sub-sample, to be used in other scripts.
output_dirname = str(path_to_data_dir_1)
pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)

serializable_rep = {"a_MoS2": a,
                    "interpolation_factors": interpolation_factors,
                    "atomic_potential_extent": atomic_potential_extent,
                    "sample_supercell_reduced_xy_dims_in_pixels": \
                    sample_supercell_reduced_xy_dims_in_pixels}
filename = output_dirname + "/sample_model_params_subset.json"
with open(filename, "w", encoding="utf-8") as file_obj:
    json.dump(serializable_rep, file_obj, ensure_ascii=False, indent=4)



# Write atomic coordinates of target sample to file.
filename = output_dirname + "/atomic_coords.xyz"
with open(filename, "w") as file_obj:
    line = "MoS2 Sample\n"
    file_obj.write(line)

    unformatted_line = "\t{:18.14f}\t{:18.14f}\t{:18.14f}\n"
    formatted_line = unformatted_line.format(*sample_unit_cell_dims)
    file_obj.write(formatted_line)

    occ = 1  # Number not used except to write file with expected format.

    Z_of_Mo = 42  # Atomic number of Mo.
    Z_of_S = 16  # Atomic number of S.

    # The RMS x-displacement of Mo atoms at room temperature. Value was taken
    # from experimental data for the RMS of the in-plane displacement in
    # Schönfeld et al., Acta Cryst. B39, 404-407 (1983).
    u_x_rms_of_Mo = 0.069

    # The RMS x-displacement of S atoms at room temperature. Value was taken
    # from experimental data for the RMS of the in-plane displacement in
    # Schönfeld et al., Acta Cryst. B39, 404-407 (1983).
    u_x_rms_of_S = 0.062

    single_species_sample_unit_cells = (Mo_sample_unit_cell, S_sample_unit_cell)
    Z_set = (Z_of_Mo, Z_of_S)
    u_x_rms_set = (u_x_rms_of_Mo, u_x_rms_of_S)
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
