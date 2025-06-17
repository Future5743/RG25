########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################

import geopandas as gpd
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
    Calculates the real-world distance (in meters) between two points based on their relative positions and pixel size.

    Parameters:
    -----------
    pos1: list
        Coordinates of the first point (relative to image/grid)

    pos2: list
        Coordinates of the second point

    pixel_size: float
        Pixel resolution in meters (i.e., ground size of one pixel)

    Returns:
    --------
    float: Distance between the two points in meters
    '''
    pixel_dist = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    return pixel_dist * pixel_size


def Miller_index(min_pos, max_coord_relative, pixel_size):
    '''
    Calculates the Miller circularity index of a crater, based on its rim and lowest point geometry.

    The index helps assess how circular a shape is. A value near 1.0 indicates a more circular shape.

    Parameters:
    -----------
    min_pos: tuple
        Coordinates of the crater's lowest point (relative to image/grid)

    max_coord_relative: list
        List of coordinates for maximum elevation points along the rim

    pixel_size: float
        Pixel resolution in meters

    Returns:
    --------
    float: Miller circularity index (rounded to 4 decimal places)
    '''

    perimeter = 0.0
    area = 0.0

    min_pos = list(min_pos)
    if 0 in min_pos:
        min_pos.remove(0)

    for i in range(len(max_coord_relative) - 1):
        # Triangle formed by min_pos, max[i], and max[i+1]
        a = distance_calculation(max_coord_relative[i], max_coord_relative[i + 1], pixel_size)
        b = distance_calculation(min_pos, max_coord_relative[i], pixel_size)
        c = distance_calculation(min_pos, max_coord_relative[i + 1], pixel_size)

        # Heron's formula for triangle area
        s = (a + b + c) / 2
        try:
            triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        except ValueError:
            triangle_area = 0.0  # In case of invalid sqrt due to floating-point precision errors

        perimeter += a
        area += triangle_area

    perimeter = round(perimeter, 6)
    area = round(area, 6)

    if perimeter == 0:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return round(circularity, 4)
