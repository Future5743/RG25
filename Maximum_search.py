########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import skimage as sk
import rasterio
import numpy as np
from shapely.geometry import Point, LineString

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################


def find_maxima(min_position, min_value, masked_image, out_transform):
    """
    Identifies the local maxima along radial profiles (every 10°) from a minimum point
    on a crater rim. Useful for crater rim profiling and analysis.

    Parameters
    ----------
    min_position : tuple
        Coordinates of the lowest point (0, x, y) in the image.

    min_value : float
        Minimum elevation value (crater bottom).

    profile_length : float
        Length of the radial profile lines.

    masked_image : np.ndarray
        3D masked image array (e.g., from a DEM with a shape like [1, height, width]).

    out_transform : affine.Affine
        Affine transform used to convert pixel coordinates to spatial coordinates.

    Returns
    -------
    lowest_point_coord : tuple
        Real-world coordinates (longitude, latitude) of the lowest point.

    min_geometry : shapely.geometry.Point
        Shapely geometry of the lowest point.

    not_enough_data : int
        Flag (1 or 0) indicating whether a profile contains too much missing data.

    max_values : list
        List to store the maximum values found along each profile.

    max_coords_relative : list
        List to store relative pixel coordinates of each maximum.

    max_coords_real : list
        List to store real-world coordinates (longitude, latitude) of each maximum.

    max_geometries : list
        List to store shapely Point geometries of each maximum.

    half_profiles_values : list
        List to store elevation values of each half-profile.

    half_profiles_coords_relative : list
        List to store relative pixel coordinates for each half-profile.

    Notes
    -----
    - Profiles are extracted every 10° around the lowest point.
    - If a profile is too short (< 4 pixels) or its max is too close to the edge,
      it is skipped.
    - If maximum == minimum, the profile is considered potentially invalid.
    """
    max_values = [0] * 36                               # Stores altitude values for highest_points
    max_coords_relative = [0] * 36                      # Stores relative coordinates of highest_points
    max_coords_real = [0] * 36                          # Stores the actual coordinates of the highest_points
    max_geometries = [0] * 36                           # Stores geometries of highest_points
    half_profiles_values = [0] * 36                     # Stores topographic profiles
    half_profiles_coords_relative = [0] * 36            # Stores the relative coordinates of points in the profile

    lowest_point_coord = None
    min_geometry = None
    not_enough_data = 0

    angle = 0  # Start angle in degrees
    x0, y0 = min_position[1], min_position[2]  # Crater lowest point in pixel coords

    height, width = masked_image.shape[1:3]

    for a in range(36):

        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Calcul du t maximal pour rester dans l'image
        t_values = []

        if dx != 0:
            if dx > 0:
                t_right = (width - 1 - y0) / dx
                t_values.append(t_right)
            else:
                t_left = -y0 / dx
                t_values.append(t_left)

        if dy != 0:
            if dy > 0:
                t_bottom = (height - 1 - x0) / dy
                t_values.append(t_bottom)
            else:
                t_top = -x0 / dy
                t_values.append(t_top)

        t_max = min(t_values)

        x1 = int(round(x0 + t_max * dy))
        y1 = int(round(y0 + t_max * dx))

        # Trace la ligne jusqu’au bord de l’image
        rr, cc = sk.draw.line(x0, y0, x1, y1)
        half_profiles_coords_relative[(a + 9) % 36] = [rr, cc]

        profile_values = masked_image[0, rr, cc]

        mask = getattr(profile_values, 'mask', np.zeros_like(profile_values, dtype=bool))

        len_without_mask = [val for val in mask if not val]

        half_profiles_values[(a + 9) % 36] = list(profile_values)

        angle += 10

        if len(len_without_mask) > 3:

            max_val = np.nanmax(profile_values)

            # Avoid using edge values as max (possible artifacts)

            while max_val in profile_values[-3:]:
                max_indices_invalid = np.where(profile_values == max_val)
                profile_values[max_indices_invalid] = -np.inf
                max_val = np.max(profile_values)

            if max_val != min_value:

                max_values[(a + 9) % 36] = round(max_val, 4)

                max_index = np.where(masked_image[0, rr, cc] == max_val)[0][0]

                max_coord = (rr[max_index], cc[max_index])
                max_coords_relative[(a + 9) % 36] = max_coord

                real_coord = rasterio.transform.xy(out_transform, *max_coord)
                max_coords_real[(a + 9) % 36] = real_coord

                mask = getattr(profile_values, 'mask', np.zeros_like(profile_values, dtype=bool))
                nan_count = 0

                for i in range(max_index):
                    if mask[i]:
                        nan_count += 1
                    else:
                        if nan_count > 1:
                            not_enough_data = 1
                            nan_count = 0

                lowest_point_coord = rasterio.transform.xy(out_transform, rr[0], cc[0])

                max_geometries[(a + 9) % 36] = Point(real_coord)

                min_geometry = Point(lowest_point_coord)

    max_values = [x for x in max_values if x != 0]
    max_coords_relative = [x for x in max_coords_relative if x != 0]
    max_coords_real = [x for x in max_coords_real if x != 0]
    max_geometries = [x for x in max_geometries if x != 0]

    return lowest_point_coord, min_geometry, not_enough_data, max_values, max_coords_relative, max_coords_real, \
           max_geometries, half_profiles_values, half_profiles_coords_relative




def horizontal_90(min_pos, masked_image, no_data_value, out_transform):
    # Pour la ligne de pixels extraite, trouver les valeurs des colonnes à droite de la position minimale dans la même
    # ligne
    # Index de la ligne où se trouve l'altitude la plus basse
    row_index = min_pos[1]
    # Extraction de la ligne à partir de la position min_pos vers la droite
    line = masked_image[0, row_index, min_pos[2]:]

    # Filtrer les valeurs "no data" de la ligne
    line_values_rg = line[line != no_data_value]

    if line_values_rg.size > 0:
        # Trouver l'altitude la plus haute dans la ligne filtrée
        max_val_right = line_values_rg.max()  # Trouver l'altitude la plus haute dans cette ligne
        max_pos_right = (row_index, min_pos[2] + np.where(line == max_val_right)[0][
            0])  # Calculer la position de l'altitude la plus haute

        # Vérifier si c'est la dernière valeur de la ligne
        derniere_val_line = np.where(line == max_val_right)[0][0]
        range_dern_val_line = derniere_val_line in {len(line) - 1, len(line) - 2, len(line) - 3}

        # Convertir les indices en coordonnées réelles
        max_x_rg, max_y_rg = rasterio.transform.xy(out_transform, max_pos_right[0], max_pos_right[1])

        # Créer un objet Point
        point_haut_horiz_90 = Point(max_x_rg, max_y_rg)

        # Ajouter le point avec l'altitude la plus basse à la liste
        # highest_points.append({'geometry': point_haut_horiz_90, 'max_altitude_right': max_val_right, 'run_id': id,
        # 'NAC_DTM_ID': nac_id})

        return max_val_right, point_haut_horiz_90



def horizontal_270(min_pos, masked_image, no_data_value, out_transform):
    row_index = min_pos[1]
    # Trouver les valeurs des colonnes à droite de la position minimale dans la même ligne
    line_lf = masked_image[0, row_index,
              :min_pos[2] + 1]  # Extraire la ligne à partir de la position min_pos vers la gauche

    # Filtrer les valeurs "no data" de la ligne
    line_values_lf = line_lf[line_lf != no_data_value]

    if line_values_lf.size > 0:
        # Trouver l'altitude la plus haute dans la ligne filtrée
        max_val_left = line_values_lf.max()  # Trouver l'altitude la plus haute dans cette ligne
        max_pos_left = (
        row_index, np.where(line_lf == max_val_left)[0][0])  # Calculer la position de l'altitude la plus haute

        # Vérifier si c'est la première valeur de la ligne
        premiere_val_line = np.where(line_lf == max_val_left)[0][0]
        range_prem_val_line = premiere_val_line in {0, 1, 2}

        # Convertir les indices en coordonnées réelles
        max_x_lf, max_y_lf = rasterio.transform.xy(out_transform, max_pos_left[0], max_pos_left[1])

        # Créer un objet Point
        point_haut_horiz_270 = Point(max_x_lf, max_y_lf)

        return max_val_left, point_haut_horiz_270



def vertical_360(min_pos, masked_image, no_data_value, out_transform):
    # Trouver les valeurs des colonnes en haut de la position minimale dans la même colonne
    col_top_index = min_pos[2]  # Index de la colonne où se trouve l'altitude la plus basse
    col_top = masked_image[0, :min_pos[1] + 1,
              col_top_index]  # Extraire la colonne à partir de la position min_pos vers le haut

    # Filtrer les valeurs "no data" de la ligne
    col_values_top = col_top[col_top != no_data_value]

    if col_values_top.size > 0:
        # Trouver l'altitude la plus haute dans la ligne filtrée
        max_val_top = col_values_top.max()  # Trouver l'altitude la plus haute dans cette ligne
        max_pos_top = (
        np.where(col_top == max_val_top)[0][0], col_top_index)  # Calculer la position de l'altitude la plus haute

        # Vérifier si c'est la première valeur de la colonne
        premiere_val_col = np.where(col_top == max_val_top)[0][0]
        range_prem_val_col = premiere_val_col in {0, 1, 2}

        # Convertir les indices en coordonnées réelles
        max_x_top, max_y_top = rasterio.transform.xy(out_transform, max_pos_top[0], max_pos_top[1])

        # Créer un objet Point
        point_haut_vert_360 = Point(max_x_top, max_y_top)

        # Ajouter le point avec l'altitude la plus basse à la liste
        # highest_points.append({'geometry': point_haut_vert_360, 'max_altitude_right': max_val_top, 'run_id': id,
        # 'NAC_DTM_ID': nac_id})

        return max_val_top, point_haut_vert_360



def vertical_180(min_pos, masked_image, no_data_value, out_transform):
    # Trouver les valeurs des lignes en dessous de la position minimale dans la même colonne
    col_bas_index = min_pos[2]  # Index de la colonne où se trouve l'altitude la plus basse
    col_bas = masked_image[0, min_pos[1]:,
              col_bas_index]  # Extraire la colonne à partir de la position min_pos vers le bas

    # Filtrer les valeurs "no data" de la colonne
    col_values_bas = col_bas[col_bas != no_data_value]

    if col_values_bas.size > 0:
        # Trouver l'altitude la plus haute dans la colonne filtrée
        max_val_bas = col_values_bas.max()  # Trouver l'altitude la plus haute dans cette colonne
        max_pos_bas = (min_pos[1] + np.where(col_bas == max_val_bas)[0][0],
                       col_bas_index)  # Calculer la position de l'altitude la plus haute

        # Vérifier si c'est la dernière valeur de la colonne
        derniere_val_col = np.where(col_bas == max_val_bas)[0][0]
        range_derniere_val_col = derniere_val_col in {len(col_bas) - 1, len(col_bas) - 2, len(col_bas) - 3}

        # Convertir les indices en coordonnées réelles
        max_x_bas, max_y_bas = rasterio.transform.xy(out_transform, max_pos_bas[0], max_pos_bas[1])

        # Créer un objet Point
        point_haut_vert_180 = Point(max_x_bas, max_y_bas)

        return max_val_bas, point_haut_vert_180



