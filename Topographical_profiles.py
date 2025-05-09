########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import matplotlib.pyplot as plt
import os
import math
import numpy as np

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################

def calcul_distance(pos1, pos2, pixel_size):
    '''
    This function compute the real distance between two points in meters.

    Entries:
        pos1: list, array or tupple         -- Relative coordinates of the first point
        pos2: list, array or tupple         -- Relative coordinates of the second point
        pixel_size: int                     -- Size of the pixel on the terrain

    Exit data:
        pixel_dist * pixel_size: float      -- Distance between the two points
    '''

    pixel_dist = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    return pixel_dist * pixel_size



def calculate_cumulative_distances(coords_x, coords_y, pixel_size_tb):
    '''
    This function compute the cumulative distance in a given profile.

    Entries:
        coords_x: array                 -- Contains the x coordinates of each point on the given profile
        coords_y: array                 -- Contains the y coordinates of each point on the given profile
        pixel_size_tb: int              -- Size of the pixel on the terrain

    Exit data:
        cumulative_distances: list      -- Contains the distance of each point with the start point
    '''

    start_point = [coords_x[0], coords_y[0]]
    return [
        calcul_distance(start_point, [coords_x[p], coords_y[p]], pixel_size_tb)
        for p in range(len(coords_x))
    ]



def process_profile(demi_profils_value, demi_profils_coords_relatives, i, pixel_size_tb, min_X):
    '''
    Process the individual profile for a given index (here to each 10°).

    Entries:
        demi_profils_value: list                 -- Contains the elevation value of each poin on the semi-profiles
        demi_profils_coords_relatives: list      -- Contains the relative coordinates of each point on the semi-profiles
        i: int                                   -- Index
        pixel_size_tb: int                       -- Size of the pixel on the terrain
        min_X: list                              -- The smallest list of x coordinates

    Exit data:
        full_profile: list                       -- Contains all the data of elevation of the entire profile
        X: list                                  -- Contains all the distance of each point with the start point
        min_X: list                              -- The smallest list of x coordinates
        limit_profil: int                        -- The index just before the index of the smallest elevation value
    '''

    # Retrieve and reverse the first half-profile, excluding the last point to avoid duplication
    reversed_demi_profil = demi_profils_value[i][::-1][:-1]

    limit_profil = len(reversed_demi_profil)

    # Retrieve the second half-profile as-is
    forward_demi_profil = demi_profils_value[i + 18]

    # Process coordinates for both halves
    x1 = demi_profils_coords_relatives[i][0][::-1][:-1]
    y1 = demi_profils_coords_relatives[i][1][::-1][:-1]
    x2 = demi_profils_coords_relatives[i + 18][0]
    y2 = demi_profils_coords_relatives[i + 18][1]

    # Concatenate coordinates
    coords_x = np.concatenate((x1, x2))
    coords_y = np.concatenate((y1, y2))

    # Calculate cumulative distances from the first point
    X = calculate_cumulative_distances(coords_x, coords_y, pixel_size_tb)

    # Keep the shortest distance list
    min_X = min(min_X, X, key=len)

    # Combine the profiles
    full_profile = reversed_demi_profil + forward_demi_profil

    # Replace non-float32 values with NaN
    full_profile = [val if isinstance(val, np.float32) else np.nan for val in full_profile]

    return full_profile, X, min_X, limit_profil



def save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off):
    '''
    This function plot and save the profiles images.

    Entries:
        full_profile: list          -- Contains all the data of elevation of the entire profile
        X: list                     -- Contains all the distance of each point with the start point
        i: int                      -- index in the for loop
        zone: str                   -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)
        crater_id: int              -- ID of the studied crater
        swirl_on_or_off: str        -- Indicate if the crater is on or off swirl

    Exit data:
        path: str                   -- The path where all the image will be saved
    '''

    plt.figure(figsize=(40, 15))
    plt.plot(X, full_profile, marker='x')
    plt.xlabel("Distance (m)")
    plt.ylabel("Altitude")
    plt.title(f'Profil topographique pour les angles {i * 10}° et {(i + 18) * 10}°')
    plt.grid(True)

    # Gestion des dossiers
    if swirl_on_or_off == 'on-swirl':
        path = f'results/RG{zone}/profils/on_swirl/{crater_id}'
        save_path = os.path.join(path, f'Profil_{i * 10}_{(i + 18) * 10}.png')
    else:
        path = f'results/RG{zone}/profils/off_swirl/{crater_id}'
        save_path = os.path.join(path, f'Profil_{i * 10}_{(i + 18) * 10}.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return path



def adjust_profile_length(profil_individuel, min_X, limit_profil):
    '''
    This function adjust the profile length to be equal to min_X.

    Entries:
        profil_individuel: list         -- Contains all the elevation of each entire profile
        min_X: list                     -- The smallest list of x coordinates
        limit_profil: int               -- The index just before the index of the smallest elevation value

    Exit data:
        profil_individuel: list         -- Contains all the elevation of each entire profile
    '''
    if len(profil_individuel) > len(min_X):
        excedant = len(profil_individuel) - len(min_X)

        if excedant % 2 == 0:
            profil_individuel = profil_individuel[excedant // 2: -excedant // 2]
        else:
            if len(profil_individuel) / 2 < limit_profil:
                profil_individuel = profil_individuel[math.ceil(excedant / 2): -excedant // 2]
            else:
                profil_individuel = profil_individuel[excedant // 2: -math.ceil(excedant / 2)]
    return profil_individuel



def calculate_average_profile(all_profiles, min_X):
    '''
    This fuction calculate the average profile by averaging values at each distance.

    /!\ It is just to visualize, it is not a real result /!\

    Entries:
        all_profiles: list          -- Contains all the elevation of all the crater's profiles
        min_X: list                 -- The smallest list of x coordinates

    Exit data:
        profil_moyen: list          -- Contains the average elevations of the crater
    '''

    profil_moyen = []
    for x in range(len(min_X)):
        colonne_i = [sous_liste[x] for sous_liste in all_profiles]
        profil_moyen.append(np.mean(colonne_i))

    return profil_moyen



def save_average_profile(profil_moyen, min_X, path):
    '''
    This fuction plot and save the average profile.

    /!\ It is just to visualize, it is not a real result /!\

    Entries:
        profil_moyen: list          -- Contains the average elevations of the crater
        min_X: list                 -- The smallest list of x coordinates
        path: str                   -- The path where the image will be saved

    Exit data:
        no exit data
    '''

    plt.figure(figsize=(40, 15))
    plt.plot(min_X, profil_moyen, marker='x')
    plt.xlabel("Distance (m)")
    plt.ylabel("Altitude")
    plt.title("Moyenne des profils topographiques")
    plt.grid(True)
    plt.savefig(path + "/Profil_moyen.png")
    plt.close()



def main(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, swirl_on_or_off, zone, crater_id):
    '''
    This function plot and save all the profiles and the average profile of one crater.
    It is using the function above.

    Entries:
        demi_profils_value: list                 -- Contains the elevation value of each poin on the semi-profiles
        demi_profils_coords_relatives: list      -- Contains the relative coordinates of each point on the semi-profiles
        pixel_size_tb: int                       -- Size of the pixel on the terrain
        swirl_on_or_off: str                     -- Indicate if the crater is on or off swirl
        zone: str                                -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)
        crater_id: int                           -- ID of the studied crater

    Exit data:
        no exit data
    '''
    all_profiles = []
    min_X = [0] * 1000  # Trouver une variable plus exacte que 1000

    # Create profiles
    for i in range(int(len(demi_profils_value) / 2)):
        # Process profile and get the corresponding distances
        full_profile, X, min_X, limit_profil = process_profile(demi_profils_value, demi_profils_coords_relatives, i,
                                                               pixel_size_tb, min_X)

        # Store the profile
        all_profiles.append(full_profile)

        # Save and plot profile
        path = save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off)

    # Adapt profiles for future averaging
    for profil_individuel in all_profiles:
        profil_individuel = adjust_profile_length(profil_individuel, min_X, limit_profil)

    # Moyennage des profils
    profil_moyen = calculate_average_profile(all_profiles, min_X)

    # Save and plot the average profile
    save_average_profile(profil_moyen, min_X, path=path)