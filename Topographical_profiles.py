########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg', selon ton système
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
            if i < len(full_profile)/2:
                while np.isnan(full_profile[i]):
                    a += 1
                    full_profile[i] = full_profile[a]
            else:
                while np.isnan(full_profile[i]):
                    a -= 1
                    full_profile[i] = full_profile[a]

    # Création de la spline
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


def trouver_pic_local_maximal(points, start, end):
    """
    Cherche le pic local de plus grande valeur entre start et end (exclus)
    """
    meilleur_index = None
    meilleur_valeur = -np.inf

    for i in range(start + 1, end - 1):  # on évite les extrémités
        if points[i] > points[i - 1] and points[i] > points[i + 1]:
            if points[i] > meilleur_valeur:
                meilleur_valeur = points[i]
                meilleur_index = i

    if meilleur_index == None :
        meilleur_index = int(np.where(points == np.max(points[start + 1: end - 1])))
    return meilleur_index


def trouver_point_droite(points, point_min):
    n_points = len(points)
    start = point_min + 1  # on inclut maintenant le voisin direct
    available = len(points) - start

    if available <= 1:
        return len(points) - 2  # Fallback sur l'avant-dernier

    n_tranche = max(1, available)
    end = min(start + n_tranche, len(points) - 1)  # exclut le dernier point

    max_val = float('-inf')
    max_idx = start
    for i in range(start, end):  # end exclu, donc max i = len(points) - 2
        if points[i] > max_val:
            max_val = points[i]
            max_idx = i

    return max_idx


def trouver_point_gauche(points, point_min):
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



def pseudo_floor(X, derivate, point_min, smooth):

    index_gauche = trouver_point_gauche(derivate, point_min)
    index_droite = trouver_point_droite(derivate, point_min)

    X_gauche = X[index_gauche]
    X_droite = X[index_droite]

    return X_gauche, X_droite


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


def save_and_plot_profile(full_profile, X, i, limit_profil, zone, crater_id, swirl_on_or_off,
                          alt_points_inner_20, full_profile_source,
                          smooth=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import os

    def build_save_path(zone, swirl_on_or_off, crater_id, smooth, i, suffix):
        base_path = f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}'
        os.makedirs(base_path, exist_ok=True)
        filename = f"Profil{'_smoothed' if smooth else ''}_{i * 10}_{(i + 18) * 10}{suffix}.png"
        return os.path.join(base_path, filename)

    indices_20 = find_alt_point_indices(full_profile_source, alt_points_inner_20)

    if len(indices_20) >= 3:
        start_idx = indices_20[0]
        end_idx = indices_20[2]
    else:
        start_idx = indices_20[0]
        end_idx = indices_20[1]

    X_slice = X[start_idx:end_idx + 1]
    if len(X_slice) <= 2:
        return f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', limit_profil - start_idx, \
               end_idx - limit_profil

    # ---------------- MÉTHODE AUTOMATIQUE ----------------
    profile_slice = full_profile[start_idx:end_idx + 1]
    second_derivative = profile_derivative(profile_slice, X_slice)
    point_min = int(np.argmin(profile_slice))
    floor_left, floor_right = pseudo_floor(X_slice, second_derivative, point_min, smooth)

    index_left = int(np.argmin(np.abs(np.array(X) - floor_left)))
    index_right = int(np.argmin(np.abs(np.array(X) - floor_right)))

    automatic_indices = [index_left, index_right]

    # ---------------- MÉTHODE INTERACTIVE ----------------
    if X[-1] >= 400:
        smooth = True
        print(f"Le profil du cratère {crater_id} à l'angle {i * 10} a été lissé")

        selected_points = []
        selected_indices = []

        fig, ax = plt.subplots(figsize=(10, 5))
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
            fig_final, ax_final = plt.subplots(figsize=(40, 15))

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

            fig_final.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth=True, i=i, suffix="_selection"))
            plt.close(fig_final)
            plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', onclick)
        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Valider la saisie')
        button.on_clicked(on_submit)
        plt.show()

        selected_indices = sorted(selected_indices)

        return f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', limit_profil - selected_indices[0], \
               selected_indices[1] - limit_profil

    # ---------------- SAUVEGARDE AUTOMATIQUE ----------------
    plt.figure(figsize=(40, 15))
    for idx in indices_20:
        if idx < len(X):
            plt.scatter(X[idx], full_profile[idx], color='red', zorder=5)

    plt.scatter(X[index_left], full_profile[index_left], color='orange', zorder=5)
    plt.scatter(X[index_right], full_profile[index_right], color='orange', zorder=5)

    plt.plot(X, full_profile, color='blue', marker='x', label='Profil topographique')
    plt.xlabel("Distance (m)")
    plt.ylabel("Altitude")
    plt.title(f'Profil topographique pour les angles {i * 10}° à {(i + 18) * 10}°')
    plt.grid(True)
    legend_elements = [
        plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Inner 20', markersize=10),
        plt.Line2D([0], [0], color='orange', marker='o', linestyle='', label='Floor delimitation', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth=False, i=i, suffix=""))
    plt.close()

    # Dérivée seconde
    idx_l = int(np.argmin(np.abs(X_slice - floor_left)))
    idx_r = int(np.argmin(np.abs(X_slice - floor_right)))

    plt.figure(figsize=(40, 15))
    plt.scatter(X_slice[idx_l], second_derivative[idx_l], color='green', zorder=5)
    plt.scatter(X_slice[idx_r], second_derivative[idx_r], color='green', zorder=5)
    plt.plot(X_slice, second_derivative, color='red', label='Seconde dérivée')
    plt.scatter(X_slice, second_derivative, color='red', s=80)
    plt.xlabel("Distance (m)")
    plt.ylabel("Seconde dérivée")
    plt.title(f'Dérivée seconde pour les angles {i * 10}° à {(i + 18) * 10}°')
    plt.grid(True)
    plt.legend()
    plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, smooth=False, i=i, suffix="_second_derivative"))
    plt.close()

    test = [f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', limit_profil - index_left, \
           index_right - limit_profil]

    if len(test) != 3:
        print(f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', limit_profil - index_left, \
               index_right - limit_profil)
        raise ValueError(f"L'un des composant n'a pas été correctement créé."
                         f"\npath {path}"
                         f"\nidx1 {idx1}"
                         f"\nidx2{dx2}")

    return f'results/RG{zone}/profils/{swirl_on_or_off}/{crater_id}', limit_profil - index_left, \
           index_right - limit_profil


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

import numpy as np


def extract_cleaned_profile(profil_coords, profil_values, no_data_value):
    """Crée un profil propre avec les NaNs et structure [x, y, z]."""
    m = len(profil_coords[0])
    profil = np.array([[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)])
    profil[profil == no_data_value] = np.nan
    return profil


def find_inner_points(demi_profil, depth_value, min_val, min_val_profil):
    """Trouve les deux points internes proches de 20% et 80% de la profondeur."""
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
    point_min = avoid_min_value_point(demi_profil, point_min, idx_min, min_val_profil)
    point_max = avoid_min_value_point(demi_profil, point_max, idx_max, min_val_profil)

    return point_min, point_max, idx_min, idx_max


def avoid_min_value_point(profile, point, idx, min_val_profil):
    """Remplace un point si sa valeur est égale à la valeur minimale brute du profil."""
    if point is None or point[2] != min_val_profil:
        return point

    next_idx = idx + 1 if idx + 1 < len(profile) else idx - 1
    if 0 <= next_idx < len(profile) and not np.isnan(profile[next_idx][2]):
        return profile[next_idx]
    return point


def process_all_inner_points(demi_profils_coords_relatives, demi_profils_value, no_data_value, depth, min_val):
    """Traite tous les profils pour extraire les points internes à 20% et 80%."""
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

        min_val_profil = np.min(profil_values_clean)

        point_min, point_max, idx_min, idx_max = find_inner_points(
            demi_profil, depth[i], min_val, min_val_profil
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
    """Génère les profils complets, les sauvegarde et extrait les indices du plancher."""
    all_profiles = []
    min_X = [0] * 1000
    crater_floor = [0] * 36

    for i in range(int(len(demi_profils_value) / 2)):
        full_profile, X, min_X, limit_profil = process_profile(
            demi_profils_value, demi_profils_coords_relatives, i, pixel_size_tb, min_X)

        all_profiles.append(full_profile)

        alt_20 = [points_inner_20[i][0], points_inner_20[i + 18][0]]
        full_profile_source = demi_profils_value[i][::-1][:-1] + demi_profils_value[i + 18]

        path, idx1, idx2 = save_and_plot_profile(full_profile, X, i, limit_profil, zone,
                                                 crater_id, swirl_on_or_off, alt_20,
                                                 full_profile_source, smooth=False)

        crater_floor[i] = idx1
        crater_floor[i + 18] = idx2

    return all_profiles, min_X, crater_floor, limit_profil, path


def main(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, swirl_on_or_off, zone, crater_id,
         no_data_value, depth, min_val):
    """
    Plot and save all the profiles and the average profile of one crater.
    """
    points_inner_20, index_inner_20 = process_all_inner_points(
        demi_profils_coords_relatives, demi_profils_value, no_data_value, depth, min_val
    )

    all_profiles, min_X, crater_floor, limit_profil, path = process_profiles_and_plot(
        demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, points_inner_20,
        zone, crater_id, swirl_on_or_off
    )

    # Ajustement des longueurs de profils pour le moyennage
    all_profiles = [adjust_profile_length(p, min_X, limit_profil) for p in all_profiles]

    # Calcul du profil moyen
    # profil_moyen = calculate_average_profile(all_profiles, min_X)

    # Sauvegarde du profil moyen
    # save_average_profile(profil_moyen, min_X, path=path)

    return crater_floor, points_inner_20, index_inner_20



def visualisation3d (masked_image, crater_id, zone, swirl_on_or_off):
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
