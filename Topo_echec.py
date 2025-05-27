########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


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
    for pair in alt_points:
        for alt in pair:
            try:
                idx = profile.index(alt)
                indices.append(idx)
            except ValueError:
                continue  # altitude pas trouvée dans ce profil
    return indices


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


def profile_derivative(full_profile, X):
    full_profile = np.array(full_profile)
    X = np.array(X)

    if not np.all(np.isfinite(full_profile)):
        index_nan = np.where(np.isnan(full_profile))

        for i in index_nan[0]:
            a = i
            if i < len(full_profile) / 2:
                while np.isnan(full_profile[i]):
                    a += 1
                    full_profile[i] = full_profile[a]
            else:
                while np.isnan(full_profile[i]):
                    a -= 1
                    full_profile[i] = full_profile[a]

    # Création de la spline
    print(f"X avant de faire l'interpolation comme une fonction{X}")
    spline = CubicSpline(X, full_profile)

    # Création de la fonction dérivée
    spline_derivative = spline.derivative(nu=2)

    # Évaluation de la fonction et de sa dérivée
    y_interp = spline(X)
    dy_dx = spline_derivative(X)

    # Tracé
    plt.figure(figsize=(10, 5))
    plt.plot(X, full_profile, 'o', label='Points d\'origine')
    plt.plot(X, y_interp, '-', label='Interpolation (fonction f)')
    plt.title('Fonction f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.close()

    return dy_dx


def build_save_path(zone, swirl_on_or_off, crater_id, smooth, i, suffix):
    base_path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
    os.makedirs(base_path, exist_ok=True)
    filename = f"Profil{'_smoothed' if smooth else ''}_{i * 10}_{(i + 18) * 10}{suffix}.png"
    return os.path.join(base_path, filename)


def plot_and_save(x_vals, y_vals, title, ylabel, label, color, marker, suffix):
    plt.figure(figsize=(40, 15))
    plt.plot(x_vals, y_vals, color=color, marker=marker, label=label)
    plt.xlabel("Distance (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth, i, suffix))
    plt.close()


def is_valid_peak(candidate_idx, slice_data, direction):
    """Vérifie la forme de pic : descente puis remontée (forme stricte)."""
    if 1 <= candidate_idx <= len(slice_data) - 2:
        if direction == "right":
            return slice_data[candidate_idx - 1] > slice_data[candidate_idx] < slice_data[candidate_idx + 1]
        elif direction == "left":
            return slice_data[candidate_idx - 1] < slice_data[candidate_idx] > slice_data[candidate_idx + 1]
    return False


def fallback_peak(slice_data, limit_offset, direction):
    """Cherche le pic le plus fort répondant aux contraintes : non aux bords, forme de pic, 3 points d'écart."""
    valid_indices = []
    for j in range(3, len(slice_data) - 3):  # évite les extrémités
        if direction == "right":
            dist = j
        else:  # gauche
            dist = limit_offset - j
        if dist >= 3 and is_valid_peak(j, slice_data, direction):
            valid_indices.append(j)

    if valid_indices:
        return max(valid_indices, key=lambda idx: slice_data[idx])
    return None  # aucun candidat ne respecte les critères


def peak_idx(second_derivative, limit_profil):
    # --- DROITE ---
    right_slice = second_derivative[limit_profil + 1:]
    right_peak_idx = None
    if len(right_slice) >= 6:
        right_peak_value = np.max(right_slice)
        right_candidates = [
            (j, val) for j, val in enumerate(right_slice)
            if abs(val - right_peak_value) <= abs(right_peak_value) * 0.03 and j >= 3
        ]
        valid_right = [
            (j, val) for j, val in right_candidates
            if is_valid_peak(j, right_slice, direction="right")
        ]
        if valid_right:
            right_peak_idx_rel = min(valid_right, key=lambda x: abs((limit_profil + 1) - x[0]))[0]
        else:
            fallback = fallback_peak(right_slice, limit_offset=0, direction="right")
            if fallback is not None:
                right_peak_idx_rel = fallback
            else:
                right_peak_idx_rel = min(len(right_slice) - 2, 3)  # fallback final safe
        right_peak_idx = limit_profil + 1 + right_peak_idx_rel
        right_peak_idx = min(right_peak_idx, len(second_derivative) - 2)
    else:
        right_peak_idx = min(limit_profil + 3, len(second_derivative) - 2)

    # --- GAUCHE ---
    left_slice = second_derivative[:limit_profil]
    left_peak_idx = None
    if len(left_slice) >= 6:
        left_peak_value = np.max(left_slice)
        left_candidates = [
            (j, val) for j, val in enumerate(left_slice)
            if abs(val - left_peak_value) <= abs(left_peak_value) * 0.03 and (limit_profil - j) >= 3
        ]
        valid_left = [
            (j, val) for j, val in left_candidates
            if is_valid_peak(j, left_slice, direction="left")
        ]
        if valid_left:
            left_peak_idx = min(valid_left, key=lambda x: abs(limit_profil - x[0]))[0]
        else:
            fallback = fallback_peak(left_slice, limit_offset=limit_profil, direction="left")
            if fallback is not None:
                left_peak_idx = fallback
            else:
                left_peak_idx = max(1, limit_profil - 3)
    else:
        left_peak_idx = max(1, limit_profil - 3)

    return right_peak_idx, left_peak_idx


def save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                          alt_points_inner_20, alt_points_inner_30, full_profile_source,
                          smooth, delimitation_graph=None):
    '''
    Trace et sauvegarde les profils topographiques avec leur dérivée seconde.

    Paramètres :
        full_profile: list            -- Profil d'altitude (lissé ou non)
        X: list                       -- Distances en mètres
        i: int                        -- Index du profil
        zone: str                     -- Zone d'étude
        crater_id: int                -- ID du cratère
        swirl_on_or_off: str         -- 'on-swirl' ou 'off-swirl'
        alt_points_inner_20: list    -- Points [min, max] pour inner_20
        alt_points_inner_30: list    -- Points [min, max] pour inner_30
        full_profile_source: list    -- Profil brut
        smooth: bool                 -- Indique si le profil est lissé
        delimitation_graph: list    -- Indices de découpage pour les profils lissés

    Retour :
        path: str                     -- Chemin du dossier de sauvegarde
        indices_30: list             -- Indices associés à inner_30
    '''

    if not smooth:
        # Étape 1 : calcul des indices spécifiques
        indices_20 = find_alt_point_indices(full_profile_source, alt_points_inner_20)
        indices_30 = find_alt_point_indices(full_profile_source, alt_points_inner_30)

        # Étape 2 : calcul de la dérivée seconde et des pics
        if len(indices_20) >= 3:
            start_idx = indices_20[0]
            end_idx = indices_20[2]
        else:
            # Fallback sécurisé
            start_idx = indices_20[0]
            end_idx = indices_20[1]

        X_slice = X[start_idx:end_idx + 1]
        profile_slice = full_profile[start_idx:end_idx + 1]
        second_derivative = profile_derivative(profile_slice, X_slice)
        limit_profil = int(np.argmin(profile_slice))
        right_peak_idx, left_peak_idx = peak_idx(second_derivative, limit_profil)
        left_global_idx = start_idx + left_peak_idx
        right_global_idx = start_idx + right_peak_idx

        # Étape 3 : tracé du profil topographique avec points verts
        plt.figure(figsize=(40, 15))
        plt.plot(X, full_profile, color='blue', marker='x', label='Profil topographique')
        plt.xlabel("Distance (m)")
        plt.ylabel("Altitude")
        plt.title(f'Profil topographique pour les angles {i * 10}° à {(i + 18) * 10}°')
        plt.grid(True)

        # Points Inner 20 & 30
        for idx in indices_20:
            if idx < len(X):
                plt.scatter(X[idx], full_profile[idx], color='red', zorder=5)
        for idx in indices_30:
            if idx < len(X):
                plt.scatter(X[idx], full_profile[idx], color='blue', zorder=5)

        # Points verts sur le profil
        if left_global_idx < len(X):
            plt.scatter(X[left_global_idx], full_profile[left_global_idx], color='green', s=200, label='Pic gauche',
                        zorder=10)
        if right_global_idx < len(X):
            plt.scatter(X[right_global_idx], full_profile[right_global_idx], color='green', s=200, label='Pic droit',
                        zorder=10)

        # Légende
        legend_elements = [
            plt.Line2D([0], [0], color='blue', marker='o', linestyle='', label='Inner 30', markersize=10),
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Inner 20', markersize=10),
            plt.Line2D([0], [0], color='green', marker='o', linestyle='', label='Pics dérivée 2', markersize=10)
        ]
        plt.legend(handles=legend_elements)

        # Sauvegarde profil
        plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth, i, suffix=""))
        plt.close()

        # Étape 4 : tracé de la dérivée seconde avec les mêmes points verts
        plt.figure(figsize=(40, 15))
        plt.plot(X_slice, second_derivative, color='red', label='Seconde dérivée')
        plt.scatter(X_slice, second_derivative, color='red', s=80)  # Montre tous les points
        plt.scatter(X_slice[left_peak_idx], second_derivative[left_peak_idx], color='green', s=200, label='Pic gauche',
                    zorder=10)
        plt.scatter(X_slice[right_peak_idx], second_derivative[right_peak_idx], color='green', s=200, label='Pic droit',
                    zorder=10)
        plt.xlabel("Distance (m)")
        plt.ylabel("Seconde dérivée")
        plt.title(f'Dérivée seconde pour les angles {i * 10}° à {(i + 18) * 10}°')
        plt.grid(True)
        plt.legend()

        # Sauvegarde dérivée seconde
        plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth, i, "_second_derivative"))
        plt.close()

        return f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', indices_20

    else:
        # Profil lissé
        plot_and_save(X, full_profile,
                      "Lissage du profil topographique",
                      "Altitude", "Profil original", "blue", "", "")

        # Dérivée seconde sur portion délimitée
        X_slice = X[delimitation_graph[0]:delimitation_graph[2]]
        profile_slice = full_profile[delimitation_graph[0]:delimitation_graph[2]]
        second_derivative = profile_derivative(profile_slice, X_slice)

        plot_and_save(X_slice, second_derivative,
                      f'Dérivée seconde issue du profil lissé pour les angles {i * 10}° à {(i + 18) * 10}°',
                      "Altitude", "Seconde dérivée", "red", "o", "_second_derivative")

        return f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', None


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


def smooth_profile(y):
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


def main(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, swirl_on_or_off, zone, crater_id,
         alt_points_inner_20, alt_point_inner_30):
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

        alt_20 = [alt_points_inner_20[i], alt_points_inner_20[i + 18]]
        alt_30 = [alt_point_inner_30[i], alt_point_inner_30[i + 18]]

        full_profile_source = demi_profils_value[i][::-1][:-1] + demi_profils_value[i + 18]

        path, indices_20 = save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                                                 alt_20, alt_30, full_profile_source, smooth=False)

        """
        full_profile_smooth = smooth_profile(full_profile)

        save_and_plot_profile(full_profile_smooth, X, i, zone, crater_id, swirl_on_or_off, alt_20, alt_30,
                              full_profile_source, smooth=True, delimitation_graph=indices_20)
        """

    # Adapt profiles for future averaging
    for profil_individuel in all_profiles:
        profil_individuel = adjust_profile_length(profil_individuel, min_X, limit_profil)

    # Moyennage des profils
    profil_moyen = calculate_average_profile(all_profiles, min_X)

    # Save and plot the average profile
    save_average_profile(profil_moyen, min_X, path=path)


def visualisation3d(masked_image, crater_id, zone, swirl_on_or_off):
    masked_band = masked_image[0]  # masked_image est de forme (1, rows, cols)

    # Grille X, Y en fonction de la forme de l'image
    rows, cols = masked_band.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Affichage en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, masked_band, cmap='terrain', linewidth=0, antialiased=False)

    ax.set_title(f'Visualisation 3D du crater {crater_id} de RG{zone}')

    path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
    save_path = os.path.join(path, f'Representation_3d_{crater_id}.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    # plt.show()
    plt.close()
