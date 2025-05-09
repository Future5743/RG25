########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import geopandas as gpd    # Import de la bibliothèque python "Geopandas". Permet de manipuler des données géographiques
import skimage as sk
import os
import shutil
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
def distance_calculation(pos1, pos2, pixel_size):
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

def Miller_index(min_pos, max_coord_relative, pixel_size):
    '''
    This function compute the Miller compactness index, used to estimate the circularity of any shape.

    Entries:
        min_pos: tuple                      -- Relative coordinates of the crater's lowest point
        max_coord_relative: list            -- Contains all the relative coordinates of the maxima on the rim
        pixel_size: float                   -- Size of the pixel on the terrain

    Exit data:
        round(circularity, 4): float        -- Index representing the circularity of the studied shape.
                                               The closer the index is to 1, the more the shape is considered circular
    '''
    perimeter = 0.0
    area = 0.0

    min_pos = list(min_pos)
    if 0 in min_pos:
        min_pos.remove(0)

    for i in range(len(max_coord_relative) - 1):
        # Triangle formé par min_pos, max[i], max[i+1]
        a = distance_calculation(max_coord_relative[i], max_coord_relative[i + 1], pixel_size)
        b = distance_calculation(min_pos, max_coord_relative[i], pixel_size)
        c = distance_calculation(min_pos, max_coord_relative[i + 1], pixel_size)

        # Formule de Héron pour la surface du triangle
        s = (a + b + c) / 2
        try:
            triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        except ValueError:
            triangle_area = 0.0  # En cas de valeurs négatives dues à l'imprécision flottante

        perimeter += a
        area += triangle_area

    # Arrondir pour éviter division par zéro avec très petits périmètres
    perimeter = round(perimeter, 6)
    area = round(area, 6)

    if perimeter == 0:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return round(circularity, 4)