######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import numpy as np

######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################
def distance_calculation(pos1, pos2, pixel_size_tb=2):
    pixel_dist_tb = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_in_meters_tb = pixel_dist_tb * pixel_size_tb

    return distance_in_meters_tb

def max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb):
    slopes = []
    for i in range(int(len(max_value) / 2)):
        # Calcul de la distance entre les points des crêtes opposées
        dist = distance_calculation(max_coord_relative[i], max_coord_relative[i + 18], pixel_size_tb)

        # Trouver la plus basse et la plus haute altitude pour les profils tous les 10 m
        low_alt = min(max_value[i], max_value[i + 18])
        high_alt = max(max_value[i], max_value[i + 18])

        # Calcul de la difference d'altitudes
        diff_alt = round(high_alt - low_alt, 4)

        # Calcul de la pente
        slope_rad = round(np.arctan(diff_alt / dist), 4)

        # Convertion en degres
        slope_deg = round(np.rad2deg(slope_rad), 4)

        slopes.append(slope_deg)

    return np.max(slopes)

def slopes_calculation(min_pos, min_value, max_value, max_coord_relative, pixel_size_tb, precision_error):
    slopes = []
    delta_slopes = []

    for point in range(len(max_value)):
        dist = distance_calculation(min_pos, max_coord_relative[point], pixel_size_tb)

        diff_alt = round(max_value[point] - min_value)

        slope_rad = round(np.arctan(diff_alt / dist), 4)

        slope_deg = round(np.rad2deg(slope_rad), 4)

        slopes.append(slope_deg)

        delta_slope = np.sqrt(
            2 * (precision_error/(max_coord_relative[point][0]-min_pos[0]))**2
            + 2 * (((max_coord_relative[point][1] - min_pos[1]) * pixel_size_tb)/((max_coord_relative[point][0]-min_pos[0])**2))**2
        )

        delta_slopes.append(delta_slope)

    return slopes, delta_slopes

