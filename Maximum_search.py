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

def Finding_maxima(min_pos, min_val, D, masked_image, out_transform, max_value, max_coord_relative, max_coord_real,
                   max_geom, line_geom, demi_profils_value, demi_profils_coords_relatives, index_maximum):

    lowest_point_coord = None
    min_geom = None

    # Initialisation de l'angle étudié pour former les profils
    angle = 0

    # Attribition de variables pour les coordonnées relatives du lowest point
    x0, y0 = min_pos[1], min_pos[2]

    # Boucle pour étudier des profils tous les 10°
    for i in range(36):

        # Convertion de l'angle en radian
        angle_rad = np.deg2rad(angle)

        # Coordonnées des points à l'extremité du profil
        x1 = int(x0 + D * np.cos(angle_rad))
        y1 = int(y0 + D * np.sin(angle_rad))

        # Ajustement du profil à masked_image
        while True:
            try:
                masked_image[0, x1, y1]
                break
            except:
                D = D * 0.99  # Réduction de la longueur du profil de 1%
                x1 = int(x0 + D * np.cos(angle_rad))
                y1 = int(y0 + D * np.sin(angle_rad))

        # Définition de la ligne du profil étudié
        rr, cc = sk.draw.line(x0, y0, x1, y1)

        demi_profils_coords_relatives.append([rr, cc])

        line_value = masked_image[0, rr, cc]  # Extraction des altitudes de la ligne

        demi_profils_value.append(list(line_value))  # On ajoute chaque demi-profil à la liste profils

        # Ajout de 10° à l'angle
        angle += 10

        # On exclue les cratères dont les lignes de profils contiennent moins de 3 pixels
        if line_value.shape[0] > 3:

            # Extraction de l'altitude maximale
            maximum = np.max(line_value)

            # Recalcul du maximum si celui-ci correspond à un des trois derniers pixels de la ligne de profil
            while maximum == line_value[-1] or maximum == line_value[-2] or maximum == line_value[-3]:
                index = np.where(line_value == maximum)

                line_value[index] = - np.inf

                maximum = np.max(line_value)

            # Exclusion des cratères où l'altitude maximale d'un profil est égale à l'altitude minimale
            # (peut correspondre à une erreur de détection de YOLOv5)
            if maximum != min_val:
                max_value.append(round(maximum, 4))

                index_max = np.where(masked_image[0, rr, cc] == maximum)

                index_maximum.append(index_max[0][0])

                max_coordinates = (rr[index_max][0], cc[index_max][0])

                max_coord_relative.append(max_coordinates)

                max_real_coordinates = rasterio.transform.xy(out_transform, max_coordinates[0],
                                                             max_coordinates[1])

                max_coord_real.append(max_real_coordinates)

                lowest_point_coord = rasterio.transform.xy(out_transform, rr[0], cc[0])
                limit_point_coord = rasterio.transform.xy(out_transform, rr[-1], cc[-1])

                # Ajout des géométries dans leur liste correspondantes
                max_geom.append(Point(max_real_coordinates[0], max_real_coordinates[1]))
                min_geom = Point(lowest_point_coord[0], lowest_point_coord[1])
                line_geom.append(LineString([lowest_point_coord, limit_point_coord]))

    return lowest_point_coord, min_geom


def horizontal_90(min_pos, masked_image, no_data_value, out_transform):
    # Pour la ligne de pixels extraite, trouver les valeurs des colonnes à droite de la position minimale dans la même ligne
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
        # highest_points.append({'geometry': point_haut_horiz_90, 'max_altitude_right': max_val_right, 'run_id': id, 'NAC_DTM_ID': nac_id})

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
        # highest_points.append({'geometry': point_haut_vert_360, 'max_altitude_right': max_val_top, 'run_id': id, 'NAC_DTM_ID': nac_id})

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



