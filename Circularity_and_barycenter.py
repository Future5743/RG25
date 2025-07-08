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


def distance_calculation(pos1, pos2):
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
    return pixel_dist


def Miller_index(max_coord_real):
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

    for i in range(len(max_coord_real)):

        x0, y0 = max_coord_real[i]

        x1, y1 = max_coord_real[(i+1) % len(max_coord_real)]

        area += (x0*y1 - x1*y0)

        perimeter += distance_calculation([x0, y0], [x1, y1])

    perimeter = round(perimeter, 6)
    area = round(abs(area) / 2, 6)

    if perimeter == 0:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    return round(circularity, 4)


def barycenter(points):
    '''
    This function compute the barycenter of the studied crater

    Parameters:
    -----------
    points: list
        Contains the points forming the geometry to be studied

    Return:
    -------
    Cx: float
        Is the x coordinate of the computed barycenter

    Cy: float
        is the y coordinate of the computed barycenter
    '''

    n = len(points)

    A = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A = A * 0.5

    if A == 0:
        raise ValueError("The polygon area is null")

    Cx /= (6 * A)
    Cy /= (6 * A)

    return Cx, Cy
