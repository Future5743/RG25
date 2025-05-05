######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import geopandas as gpd    # Import de la bibliothèque python "Geopandas". Permet de manipuler des données géographiques

import skimage as sk

import os

import shutil

import rasterio

from rasterio.mask import mask

import numpy as np

from shapely.geometry import Point, LineString, Polygon

from tqdm import tqdm

######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################
def distance_calculation(pos1, pos2, pixel_size_tb=2):
    pixel_dist_tb = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_in_meters_tb = pixel_dist_tb * pixel_size_tb

    return distance_in_meters_tb

def Miller_index(min_pos, max_coord_relative, pixel_size_tb):
    # Calcul du perimetre et de l'aire du polygone

    perimeter = 0
    area = 0

    for i in range(len(max_coord_relative) - 1):
        # Calcul de distance pour le perimetre et la formule de Heron
        da = distance_calculation(max_coord_relative[i], max_coord_relative[i + 1], pixel_size_tb)
        db = distance_calculation(min_pos, max_coord_relative[i], pixel_size_tb)
        dc = distance_calculation(min_pos, max_coord_relative[i + 1], pixel_size_tb)

        # Formule de Heron
        p = (da + db + dc) / 2

        S = np.sqrt(p * (p - da) * (p - db) * (p - dc))

        # Perimetre
        perimeter += da

        # Aire
        area += S

    perimeter = round(perimeter, 2)
    area = round(area, 2)

    # Calcul de l'indice de circularité de Miller
    circularity = (4 * np.pi * area) / perimeter ** 2

    return circularity