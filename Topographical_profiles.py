########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import messagebox

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


def find_alt_point_indices(profile, alt_points):
    '''
    Trouve les indices des altitudes spécifiques dans un profil.

    Entrées :
        profile: list de float -- le profil d'altitudes d'origine (non nettoyé)
        alt_points: list       -- liste de [min, max] pour chaque demi-profil

    Sortie :
        indices: list d'indices où les altitudes correspondent
    '''
    indices = []
    alt_1 = alt_points[0][-1]
    idx1 = np.where(profile == alt_1)[0]

    alt_2 = alt_points[1][-1]
    idx2 = np.where(profile == alt_2)[0]

    return [int(idx1), int(idx2)]


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
    full_profile = demi_profils_value[i][::-1][:-1] + demi_profils_value[i + 18]

    # Replace non-float32 values with NaN
    full_profile = [val if isinstance(val, np.float32) else np.nan for val in full_profile]

    return full_profile, X, min_X


def profile_derivative(full_profile, X):
    '''
    This function derive twice a profile.

    Entries:
        full_profile: list              -- Contains the elevations of the profile
        X: list                         -- Is the x-axis

    Exit data:
        dy_dx: list                     -- Contains the values of the profile second derivative
    '''

    full_profile = np.array(full_profile)
    X = np.array(X)

    if not np.all(np.isfinite(full_profile)):
        index_nan = np.where(np.isnan(full_profile))

        for i in index_nan[0]:
            a = i
            if i < len(full_profile)/2:
                while np.isnan(full_profile[i]):
                    a += 1
                    full_profile[i] = full_profile[a]
            else:
                while np.isnan(full_profile[i]):
                    a -= 1
                    full_profile[i] = full_profile[a]

    # Sline creation
    spline = CubicSpline(X, full_profile)

    # Derivative function creation
    spline_derivative = spline.derivative(nu=2)

    # Evaluation of the function and its derivative
    y_interp = spline(X)
    dy_dx = spline_derivative(X)

    return dy_dx


def trouver_point_droite(points, point_min):
    '''
    This function find and return the index of the point which will be considered as the delimitation of the crater
    floor on the right side of the profile.

    To be selected, the point must meet certain criteria:
        * it can not be located on the extremes of the semi profile (to avoid edge effects and the lowest point of the
          semi-profile)
        * it have to be the highest value following the previous criteria

    Entries:
        points: list                -- Contains all the value of the second derivative
        point_min: int              -- Is the index of the lowest point on the profile

    Exit data:
        max_idx: int                -- Is the index of the point selected, considered as the right delimitation of the
                                       crater floor
    '''

    start = point_min + 1
    available = len(points) - start

    if available <= 1:
        return len(points) - 2

    n_tranche = max(1, available)
    end = min(start + n_tranche, len(points) - 1)

    max_val = float('-inf')
    max_idx = start
    for i in range(start, end):
        if points[i] > max_val:
            max_val = points[i]
            max_idx = i

    return max_idx


def trouver_point_gauche(points, point_min):
    '''
    This function find and return the index of the point which will be considered as the delimitation of the crater
    floor on the left side of the profile.

    To be selected, the point must meet certain criteria:
        * it can not be located on the extremes of the semi profile (to avoid edge effects and the lowest point of the
          semi-profile)
        * it have to be the highest value following the previous criteria

    Entries:
        points: list                -- Contains all the value of the second derivative
        point_min: int              -- Is the index of the lowest point on the profile

    Exit data:
        max_idx: int                -- Is the index of the point selected, considered as the left delimitation of the
                                       crater floor
    '''

    end = point_min  # on inclut maintenant le voisin direct
    available = end - 1  # on veut exclure points[0]

    if available < 1:
        return 1  # Fallback : deuxième point minimum

    n_tranche = max(1, available)
    start = max(1, end - n_tranche)  # commence à 1 pour ne pas inclure points[0]

    max_val = float('-inf')
    max_idx = start
    for i in range(start, end):  # end exclu
        if points[i] > max_val:
            max_val = points[i]
            max_idx = i

    return max_idx


def pseudo_floor(X, derivate, point_min):
    '''
    This function compute the potential crater floor.

    Entries:
        X: list                         -- Is the x-axis
        derivate: list                  -- Contains all the value of the second derivative
        point_min: int                  -- Is the index of the lowest point on the profile

    Exit data:
         X_gauche: float                -- Is the index of the point selected at the left considered as the delimitation
                                           of the crater floor
         X_droite: float                -- Is the index of the point selected at the right considered as the
                                           delimitation of the crater floor
    '''

    index_gauche = trouver_point_gauche(derivate, point_min)
    index_droite = trouver_point_droite(derivate, point_min)

    X_gauche = X[index_gauche]
    X_droite = X[index_droite]

    return X_gauche, X_droite


def build_save_path(zone, swirl_on_or_off, crater_id, i, suffix):
    '''
    This function build and save a path to store a figure.

    Entries:
        zone: str                                           -- Indicates the crater's zone of study (can be 1, 2, 3, 4,
                                                               5, 6 or 7)
        swirl_on_or_off: str                                -- Indicates if the crater is on or off swirl
        crater_id: int                                      -- ID of the studied crater
        i: int                                              -- Is the loop iteration. It helps to know at which angle
                                                               the profile is
        suffix: str                                         -- Indicates if the figure is a second derivative or not

    Exit data:
        os.path.join(base_path, filename): str              -- Is the path where the figure will be stored
    '''

    base_path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
    os.makedirs(base_path, exist_ok=True)
    filename = f"Profil_{i * 10}_{(i + 18) * 10}{suffix}.png"
    return os.path.join(base_path, filename)


def save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                          alt_points_inner_20):
    '''
    This function plot and save a profile and its second derivative.
    It also computes an estimation of the crater floor.
    If the profile has a length greater than 400m, the estimation of the crater floor will be interactive, else it is
    completely automated.

    how to use the interactive mode:
        * The interactive mode will open automatically if the crater profile ahs a length greater tan 400m
        * It uses a matplotlib interface
        * Two red dots are displayed in addition to the profile : they are the points that the automatic algorithm
          would have computed for the profile
        * Two options are available:
            ** Automatically calculated red dots suit you : just clisk on the "Valider la saisie" button
            ** Automatically calculated red dots don't suit you : click two points on the profile to choose better
               points to delimit the crater floor (the clicked points will appear in orange), then click on the "Valider
               la saisie" button
            ** in general, 18 profiles will be plotted (but not always)

    Entries:
        full_profile: list                                           -- Contains all the elevation values of the profile
        X: list                                                      -- Is the x-axis
        i: int                                                       -- Is teh iteration of the loop.
        limit_profil: int                                            -- limit_profile + 1 is the index of the lowest
                                                                        point value
        zone: str                                                    -- Indicates the crater's zone of study (can be 1,
                                                                        2, 3, 4, 5, 6 or 7)
        crater_id: int                                               --  ID of the studied crater
        swirl_on_or_off: str                                         -- Indicates if the crater is on or off swirl
        alt_points_inner_20: list                                    -- Indicates the point at an elevation value of 20%
                                                                        of the total depht
        full_profile_source: list                                    -- ???

    Exit data:
        limit_profil - index_left: int                               -- Is the index in the semi-profile of the point
            or limit_profil - selected_indices[0]: int                  considered as the left delimitation of the
                                                                        crater floor
        index_right - limit_profil: int                              -- Is the index in the semi-profile of the point
            or selected_indices[1] - limit_profil: int                  considered as the right delimitation of the
                                                                        crater floor
    '''

    limit_profil = np.where(full_profile == np.nanmin(full_profile))[0]

    if len(limit_profil) > 1:
        limit_profil = limit_profil[0]

    limit_profil = int(limit_profil)

    def build_save_path(zone, swirl_on_or_off, crater_id, i, suffix):
        base_path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
        os.makedirs(base_path, exist_ok=True)
        filename = f"Profil_{i * 10}_{(i + 18) * 10}{suffix}.png"
        return os.path.join(base_path, filename)

    indices_20 = find_alt_point_indices(full_profile, alt_points_inner_20)

    start_idx = indices_20[0]
    end_idx = indices_20[1]


    X_slice = X[start_idx:end_idx + 1]
    if len(X_slice) <= 2:
        return limit_profil - start_idx, end_idx - limit_profil

    # ---------------- MÉTHODE AUTOMATIQUE ----------------
    profile_slice = full_profile[start_idx:end_idx + 1]
    second_derivative = profile_derivative(profile_slice, X_slice)

    point_min = np.where(X_slice == X[limit_profil])[0]

    if len(point_min) == 1:
        point_min = int(point_min)
    else:
        point_min = int(point_min[0])

    floor_left, floor_right = pseudo_floor(X_slice, second_derivative, point_min)

    index_left = int(np.argmin(np.abs(np.array(X) - floor_left)))
    index_right = int(np.argmin(np.abs(np.array(X) - floor_right)))

    automatic_indices = [index_left, index_right]

    # ---------------- MÉTHODE INTERACTIVE ----------------

    selected_points = []
    selected_indices = []

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(X, full_profile, color='blue', marker='x', label='Profil topographique')
    ax.set_title(f'Sélectionnez deux points pour le cratère {crater_id} (angle {i * 10}°)')
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude")
    ax.grid(True)

    # Affichage des points automatiques (rouges)
    for idx in automatic_indices:
        ax.scatter(X[idx], full_profile[idx], color='red', s=150, zorder=5)

    def onclick(event):
        if len(selected_points) < 2 and event.inaxes == ax:
            idx = np.argmin(np.abs(np.array(X) - event.xdata))
            if idx not in selected_indices:
                x_real = X[idx]
                y_real = full_profile[idx]
                selected_indices.append(idx)
                selected_points.append((x_real, y_real))
                ax.scatter(x_real, y_real, color='orange', s=150, zorder=10)
                fig.canvas.draw()

    def on_submit(event):
        nonlocal selected_indices

        # Si aucun clic → on garde les indices automatiques
        if len(selected_indices) < 2:
            print("Aucun point cliqué, on garde les indices automatiques :", automatic_indices)
            selected_indices = automatic_indices
        else:
            print("Indices sélectionnés :", selected_indices)

        # Profil complet final avec légende
        fig_final, ax_final = plt.subplots(figsize=(20, 7))

        for idx in indices_20:
            if idx < len(X):
                ax_final.scatter(X[idx], full_profile[idx], color='red', zorder=5)

        for idx in selected_indices:
            ax_final.scatter(X[idx], full_profile[idx], color='orange', zorder=5)

        ax_final.plot(X, full_profile, color='blue', marker='x', label='Profil topographique')
        ax_final.set_xlabel("Distance (m)")
        ax_final.set_ylabel("Altitude")
        ax_final.set_title(f'Profil topographique pour les angles {i * 10}° à {(i + 18) * 10}°')
        ax_final.grid(True)
        legend_elements = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Inner 20', markersize=10),
            plt.Line2D([0], [0], color='orange', marker='o', linestyle='', label='Floor delimitation',
                       markersize=10)
        ]
        ax_final.legend(handles=legend_elements)

        fig_final.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix=""))
        plt.close(fig_final)
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
    button = Button(ax_button, 'Valider la saisie')
    button.on_clicked(on_submit)
    plt.show()

    selected_indices = sorted(selected_indices)

    # ---------------- SAUVEGARDE AUTOMATIQUE ----------------

    # Dérivée seconde
    idx_l = int(np.argmin(np.abs(X_slice - floor_left)))
    idx_r = int(np.argmin(np.abs(X_slice - floor_right)))

    plt.figure(figsize=(20, 7))
    plt.scatter(X_slice[idx_l], second_derivative[idx_l], color='green', zorder=5)
    plt.scatter(X_slice[idx_r], second_derivative[idx_r], color='green', zorder=5)
    plt.plot(X_slice, second_derivative, color='red', label='Seconde dérivée')
    plt.scatter(X_slice, second_derivative, color='red', s=80)
    plt.xlabel("Distance (m)")
    plt.ylabel("Seconde dérivée")
    plt.title(f'Dérivée seconde pour les angles {i * 10}° à {(i + 18) * 10}°')
    plt.grid(True)
    plt.legend()
    plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix="_second_derivative"))
    plt.close()

    return limit_profil - selected_indices[0], selected_indices[1] - limit_profil


def adjust_profile_length(profil_individuel, min_X):
    '''
    This function adjust the profile length to be equal to min_X.

    Entries:
        profil_individuel: list         -- Contains all the elevation of each entire profile
        min_X: list                     -- The smallest list of x coordinates
        limit_profil: int               -- The index just before the index of the smallest elevation value

    Exit data:
        profil_individuel: list         -- Contains all the elevation of each entire profile
    '''
    limit_profil = np.where(full_profile == np.nanmin(full_profile))[0]

    if len(limit_profil) > 1:
        limit_profil = limit_profil[0]

    limit_profil = int(limit_profil)

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

    plt.figure(figsize=(20, 7))
    plt.plot(min_X, profil_moyen, marker='x')
    plt.xlabel("Distance (m)")
    plt.ylabel("Altitude")
    plt.title("Moyenne des profils topographiques")
    plt.grid(True)
    plt.savefig(path + "/Profil_moyen.png")
    plt.close()


def smooth_profile(y):
    '''
    This functon smmoth a profile.

    Entries:
        y: list                     -- Contains the elevations of the profile

    Exit data:
        y_smmoth: list              -- Contains the elevations of the smoothed profile
    '''

    y = np.asarray(y)

    window_length = min(21, len(y) - 1)

    if window_length % 2 == 0:
        window_length -= 1

    polyorder = 2

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        index_nan = np.where(np.isnan(y))

        for i in index_nan[0]:
            a = i
            if i < len(y) / 2:
                while np.isnan(y[i]):
                    a += 1
                    y[i] = y[a]
            else:
                while np.isnan(y[i]):
                    a -= 1
                    y[i] = y[a]

    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 != 0 else len(y) - 1
        if window_length < polyorder + 2:
            raise ValueError(f"Taille de y trop petite ({len(y)}) pour appliquer un filtre de Savitzky-Golay")

    polyorder = min(polyorder, window_length - 1)

    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)

    return y_smooth


def extract_cleaned_profile(profil_coords, profil_values, no_data_value):
    '''
    This function create a clean profile with NaNa and a [x, y, z] structure.

    Entries:
        profil_coords: list                     -- Contains the coordinates of the profile points
        profil_values: list                     -- Contains the elevation of the profile point
        no_data_value: ???                      -- ???

    Exit data:
        profil: list                            -- Is the clean profile
    '''

    m = len(profil_coords[0])
    profil = np.array([[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)])
    profil[profil == no_data_value] = np.nan
    return profil


def find_inner_points(demi_profil, depth_value, min_val):
    '''
    This function compute the two inner points near 20% and 80% of the total depht.

    Entries:
        demi_profil: list                   -- Contains the coordinates and elevation of each points on the semi-profile
        depth_value: float                  -- Is the value of the semi-profile depth
        min_val: float                      -- IS the elevation of the lowest point of the semi-profile

    Exit data:
        point_min: list                     -- Is the point near 80%
        point_max: list                     -- Is the points near 20%
        idx_min: int                        -- Is the index in the semi-profile of the point near 20%
        idx_max: int                        -- Is the index in the semi-profile of teh point near 80%
    '''
    alt_min = 0.2 * depth_value + min_val
    alt_max = 0.8 * depth_value + min_val

    point_min = point_max = None
    idx_min = idx_max = -1
    dist_min = dist_max = np.inf

    for idx, (_, _, z) in enumerate(demi_profil):
        if np.isnan(z):
            continue
        if abs(z - alt_min) < dist_min:
            dist_min, point_min, idx_min = abs(z - alt_min), demi_profil[idx], idx
        if abs(z - alt_max) < dist_max:
            dist_max, point_max, idx_max = abs(z - alt_max), demi_profil[idx], idx

    # S'assurer de l'ordre
    if idx_min > idx_max:
        idx_min, idx_max = idx_max, idx_min
        point_min, point_max = point_max, point_min

    # Éviter les points égaux à la valeur minimale brute
    point_min = avoid_min_value_point(demi_profil, point_min, idx_min, min_val)
    point_max = avoid_min_value_point(demi_profil, point_max, idx_max, min_val)

    return point_min, point_max, idx_min, idx_max


def avoid_min_value_point(profile, point, idx, min_val):
    '''
    This function replace a point if its value is equal to the minimum value of the profile.

    Entries:
        profile: list                               -- Contains the coordinates and teh elevation of each point on the
                                                       profile
        point: list                                 -- Is teh studied point
        idx: int                                    -- Is the index of the studied point
        min_val: float                              -- Is the elevation of the lowest point

    Exit data:
        point: list                                 -- Can be either the original point or another point that is close
            or profile[next_idx]: list                 to teh 20% or the 80% of the total depht
    '''

    if point is None or point[2] != min_val:
        return point

    next_idx = idx + 1 if idx + 1 < len(profile) else idx - 1
    if 0 <= next_idx < len(profile) and not np.isnan(profile[next_idx][2]):
        return profile[next_idx]
    return point


def process_all_inner_points(demi_profils_coords_relatives, demi_profils_value, no_data_value, depth, min_val):
    '''
    This function processes all the profiles to extract inner points at 20% and 80%.

    Entries:
        demi_profils_coords_relatives: list             -- Contains the relatives coordinates of each point of each
                                                           semi-profile
        demi_profils_value: list                        -- Contains the elevation of each points of each semi-profile
        no_data_value: ???                              -- ???
        depth: list                                     -- Contains the depht of esch profile
        min_val: float                                  -- Is the elevation of the lowest point of the crater

    Exit data:
        points_inner_20: list                           -- Contains the coordinates and elevation of the inner points
                                                           selected
        index_inner_20:list                             -- Contains teh index in the semi-profile of all the inner
                                                           points selected
    '''

    points_inner_20 = []
    index_inner_20 = []

    for i, (coords, values) in enumerate(zip(demi_profils_coords_relatives, demi_profils_value)):
        demi_profil = extract_cleaned_profile(coords, values, no_data_value)
        profil_values_clean = np.array(values)
        profil_values_clean = profil_values_clean[~np.isnan(profil_values_clean)]
        if len(profil_values_clean) == 0:
            print(f"Profil {i} vide après nettoyage.")
            points_inner_20.append([None, None])
            index_inner_20.append([-1, -1])
            continue

        point_min, point_max, idx_min, idx_max = find_inner_points(
            demi_profil, depth[i], min_val
        )

        if point_min is None or point_max is None:
            print(f"Profil {i}: point_inner_min ou point_inner_max est None")
            points_inner_20.append([None, None])
            index_inner_20.append([-1, -1])
        else:
            points_inner_20.append([point_min, point_max])
            index_inner_20.append([idx_min, idx_max])

    return points_inner_20, index_inner_20


def process_profiles_and_plot(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb,
                              points_inner_20, zone, crater_id, swirl_on_or_off):
    '''
    This function generate the profiles, save them and extract the index of the delimitations of the crater floor in the
    given profile.

    Entries:
        demi_profils_value: list                            -- Contains the elevation of each points of each
                                                               semi-profile
        demi_profils_coords_relatives: list                 -- Contains the relatives coordinates of each point of each
                                                               semi-profile
        pixel_size_tb: int                                  -- Size of the pixel on the terrain
        points_inner_20: list                               -- Contains all the inner points for each semi-profile (2
                                                               points per semi-profile)
        zone: str                                           -- Indicate the crater's zone of study (can be 1, 2, 3, 4,
                                                               5, 6 or 7)
        crater_id: int                                      -- ID of the studied crater
        swirl_on_or_off: str                                -- Indicate if the crater is on or off swirl

    Exit data:
        crater_floor: list                                  -- Contains all the points that where selected to delimit
                                                               the crater floor
    '''

    min_X = [0] * 1000
    crater_floor = [0] * 36
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.subplots_adjust(bottom=0.3)

    crater_morph = None

    colors = ['black', 'silver', 'red', 'saddlebrown', 'orange', 'bisque',  'gold', 'darkgoldenrod', 'yellow',
              'greenyellow', 'green', 'lime', 'turquoise', 'blue', 'skyblue', 'purple', 'plum', 'pink']

    for i in range(int(len(demi_profils_value) / 2)):

        full_profile, X, min_X = process_profile(
            demi_profils_value, demi_profils_coords_relatives, i, pixel_size_tb, min_X)

        plt.figure(figsize=(20, 7))
        plt.plot(X, full_profile, color='b', label='Profil')
        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation")
        plt.title(f'Profil topographique pour les angles {i * 10}° à {(i + 18) * 10}° du cratère {crater_id}')
        plt.grid(True)
        plt.legend()
        plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix=""))
        plt.close()

        limit_profil = np.where(full_profile == np.nanmin(full_profile))[0]

        if len(limit_profil) > 1:
            limit_profil = limit_profil[0]

        limit_profil = int(limit_profil)

        X_test = [indice - limit_profil for indice in range(len(X))]

        def set_morph_to_Bowl(event):
            nonlocal crater_morph
            crater_morph = "Bowl-shaped"
            print(f"Le cratère {crater_id} de la zone RG{zone} est un bowl-shaped.")
            plt.close()

        def set_morph_to_Flat(event):
            nonlocal crater_morph
            crater_morph = "Flat-floored"
            print(f"Le cratère {crater_id} de la zone RG{zone} est un flat-floored.")
            plt.close()

        def set_morph_to_Mound(event):
            nonlocal crater_morph
            crater_morph = "With a mound"
            print(f"Le cratère {crater_id} de la zone RG{zone} a un mound.")
            plt.close()

        def set_morph_to_Unknown(event):
            nonlocal crater_morph
            crater_morph = "Unknown"
            print(f"Le cratère {crater_id} de la zone RG{zone} est un unknown.")
            plt.close()

        ax.plot(X_test, full_profile, color=colors[i], marker='', label='Profil topographique', linewidth=0.5)

    button_axes = [
        plt.axes([0.05 +i*0.2, 0.1, 0.15, 0.075])
        for i in range(4)
    ]

    button1 = Button(button_axes[0], 'Bowl-shaped')
    button2 = Button(button_axes[1], 'Flat-floored')
    button3 = Button(button_axes[2], 'With a mound')
    button4 = Button(button_axes[3], 'Unknown')

    button1.on_clicked(set_morph_to_Bowl)
    button2.on_clicked(set_morph_to_Flat)
    button3.on_clicked(set_morph_to_Mound)
    button4.on_clicked(set_morph_to_Unknown)

    ax.set_title(f'Sélectionnez deux points pour le cratère {crater_id}')
    ax.set_xlabel("Distance en point par rapport au point le plus bas")
    ax.set_ylabel("Altitude")
    ax.grid(True)
    plt.show()

    if crater_morph != "Bowl-shaped":
        for i in range(int(len(demi_profils_value) / 2)):
            full_profile, X, min_X = process_profile(
                demi_profils_value, demi_profils_coords_relatives, i, pixel_size_tb, min_X)

            alt_20 = [points_inner_20[i][0], points_inner_20[i + 18][0]]

            idx1, idx2 = save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                                               alt_20)

            crater_floor[i] = idx1
            crater_floor[i + 18] = idx2

        if crater_morph == "Unknown":

            def on_oui():
                # Afficher les boutons supplémentaires si "Oui" est pressé
                btn_bowl.pack(pady=5)
                btn_flat.pack(pady=5)
                btn_mound.pack(pady=5)

            def on_non():
                root.destroy()  # Ferme la fenêtre

            def on_choice(choice):
                crater_morph = choice

                if choice == "Bowl-shaped":
                    crater_floor = [0] * 36
                root.destroy()  # Ferme la fenêtre après le choix

            # Création de la fenêtre principale
            root = tk.Tk()
            root.title("Choix de l'utilisateur")

            # Question en haut
            label_question = tk.Label(root, text="Avez-vous changé d'avis quant à la morphologie du cratère ?",
                                      font=("Helvetica", 12))
            label_question.pack(pady=15)

            # Boutons "Oui" et "Non"
            btn_oui = tk.Button(root, text="Oui", width=20, command=on_oui)
            btn_oui.pack(pady=5)

            btn_non = tk.Button(root, text="Non", width=20, command=on_non)
            btn_non.pack(pady=5)

            # Boutons supplémentaires (cachés au début)
            btn_bowl = tk.Button(root, text="Bowl-shaped", width=20, command=lambda: on_choice("Bowl-shaped"))
            btn_flat = tk.Button(root, text="Flat-floored", width=20, command=lambda: on_choice("Flat-floored"))
            btn_mound = tk.Button(root, text="With a mound", width=20, command=lambda: on_choice("With a mound"))

            # Lancement de la boucle principale
            root.mainloop()

    return crater_floor, crater_morph



def main(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, swirl_on_or_off, zone, crater_id,
         no_data_value, depth, min_val):
    '''
    This function plot and save profiles, but also do an estimation of the crater floor for each profile, and compute
    the two inner points near 20% and 80% of the total depht per profile.

    Entries:
        demi_profils_value: list                            -- Contains the elevation of each points of each
                                                               semi-profile
        demi_profils_coords_relatives: list                 -- Contains the relatives coordinates of each point of
                                                               each semi-profile
        pixel_size_tb: int                                  -- Size of the pixel on the terrain
        swirl_on_or_off: str                                -- Indicate if the crater is on or off swirl
        zone: str                                           -- Indicate the crater's zone of study (can be 1, 2, 3, 4,
                                                               5, 6 or 7)
        crater_id: int                                      -- ID of the studied crater
        no_data_value: ???                                  -- ???
        depth: list                                         -- Contains the depht of each profile
        min_val: float                                      -- Is the elevation of the lowest point of the crater

    Exit data:
        crater_floor: list                                  -- Contains all the points that where selected to delimit
                                                               the crater floor
        points_inner_20: list                               -- Contains the coordinates and elevation of the inner
                                                               points selected
        index_inner_20: list                                -- Contains teh index in the semi-profile of all the inner
                                                               points selected
    '''

    points_inner_20, index_inner_20 = process_all_inner_points(
        demi_profils_coords_relatives, demi_profils_value, no_data_value, depth, min_val
    )

    crater_floor, crater_morph = process_profiles_and_plot(
        demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, points_inner_20,
        zone, crater_id, swirl_on_or_off
    )

    # Ajustement des longueurs de profils pour le moyennage
    # all_profiles = [adjust_profile_length(p, min_X, limit_profil) for p in all_profiles]

    # Calcul du profil moyen
    # profil_moyen = calculate_average_profile(all_profiles, min_X)

    # Sauvegarde du profil moyen
    # save_average_profile(profil_moyen, min_X, path=path)

    return crater_floor, points_inner_20, index_inner_20, crater_morph


def visualisation3d(masked_image, crater_id, zone, swirl_on_or_off):
    '''
    This function plot and save a 3D model of the studied crater.

    Entries:
        masked_image: ???                       -- ???
        crater_id: int                          -- ID of the studied crater
        zone: str                               -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)
        swirl_on_or_off: str                    -- Indicate if the crater is on or off swirl

    Exit data:
        No exit data
    '''
    masked_band = masked_image[0]  # masked_image est de forme (1, rows, cols)

    # Grille X, Y en fonction de la forme de l'image
    rows, cols = masked_band.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Affichage en 3D
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, masked_band, cmap='terrain', linewidth=0, antialiased=False)

    ax.set_title(f'Visualisation 3D du crater {crater_id} de RG{zone}')

    path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
    save_path = os.path.join(path, f'Representation_3d_{crater_id}.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    # plt.show()
    plt.close()
