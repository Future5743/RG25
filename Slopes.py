########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from shapely.geometry import LineString

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################

def distance_calculation(pos1, pos2, pixel_size):
    '''
    Calculate the real distance (in meters) between two points given their relative coordinates and pixel size.

    Parameters:
    -----------
    pos1: list, array, or tuple
        Relative coordinates of the first point (e.g., [x1, y1])

    pos2: list, array, or tuple
        Relative coordinates of the second point (e.g., [x2, y2])

    pixel_size: int or float
        Size of one pixel on the terrain in meters

    Returns:
    --------
    float
        The real distance between the two points (meters)
    '''
    pixel_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return pixel_dist * pixel_size


def max_crater_slopes_calculation(max_values, max_coords_relative, pixel_size):
    '''
    Calculate the maximum slope between opposite sides of a crater rim.

    Parameters:
    -----------
    max_values: list or array
        Elevations of the maxima points on the crater rim

    max_coords_relative: list or array
        Relative coordinates of the maxima points on the crater rim

    pixel_size: int or float
        Size of one pixel on the terrain in meters

    Returns:
    --------
    float
        Maximum slope (in degrees) between two opposite rim points
    '''
    slopes = []
    # Assuming the rim maxima are arranged so that points i and i+18 are opposite
    for i in range(len(max_values) // 2):
        dist = distance_calculation(max_coords_relative[i], max_coords_relative[i + 18], pixel_size)
        delta_z = abs(max_values[i] - max_values[i + 18])
        slope_deg = np.rad2deg(np.arctan(delta_z / dist))
        slopes.append(round(slope_deg, 4))
    return np.max(slopes)


def slopes_px2px(slopes_list, slopes_uncertainties, index_min_inner, index_max_inner, demi_profile, pixel_size, dz):
    '''
    Calculate the average slope using the Stopar et al., 2017 method by computing slopes between adjacent points within a given profile range.
    Uncertainties are also computed for each slope.

    Parameters:
    -----------
    slopes_list: list
        List to append slopes between adjacent points

    slopes_uncertainties: list
        List to append uncertainties for each slope

    index_min_inner: int
        Starting index of the profile segment to analyze

    index_max_inner: int
        Ending index of the profile segment to analyze

    demi_profile: list of lists or arrays
        List of points defining the semi-profile; each point contains [x, y, elevation]

    pixel_size: int or float
        Size of one pixel on the terrain in meters

    dz: float
        Vertical uncertainty (elevation measurement error)

    Returns:
    --------
    tuple of floats
        mean_slope_px: average slope (degrees) over the profile segment

        mean_uncertainty: average uncertainty associated with the slope calculation
    '''
    for j in range(index_min_inner, index_max_inner):
        pt1 = demi_profile[j]
        pt2 = demi_profile[j + 1]

        dist = distance_calculation(pt1[:-1], pt2[:-1], pixel_size)
        if dist == 0:
            continue

        depth = pt2[-1] - pt1[-1]

        slope = round(np.rad2deg(np.arctan(depth / dist)), 2)
        slopes_list.append(slope)

        # Compute uncertainty for the slope between pt1 and pt2
        slope_uncertainties(slopes_uncertainties, pt1, pt2, dist, pixel_size, dz)

    if slopes_list:
        mean_slope_px = round(np.nanmean(slopes_list), 2)
        mean_uncertainty = round(np.sqrt(np.nansum(np.array(slopes_uncertainties) ** 2)) / len(slopes_uncertainties), 2)
    else:
        mean_slope_px = np.nan
        mean_uncertainty = np.nan

    return mean_slope_px, mean_uncertainty


def slope_uncertainties(uncertainties_list, point_1, point_2, dist, pixel_size, dz):
    '''
    Calculate the uncertainty associated with a slope computed between two points.

    Parameters:
    -----------
    uncertainties_list: list
        List to append the calculated slope uncertainty

    point_1: list or array
        Coordinates and elevation of the first point [x, y, z]

    point_2: list or array
        Coordinates and elevation of the second point [x, y, z]

    dist: float
        Distance between the two points (meters)

    pixel_size: int or float
        Size of one pixel on the terrain (meters)

    dz: float
        Vertical uncertainty (elevation measurement error)

    Returns:
    --------
    None
        The function appends the uncertainty to uncertainties_list.
    '''
    x = point_1[0] - point_2[0]
    y = point_1[1] - point_2[1]
    z = point_1[2] - point_2[2]

    delta_slope = (1 / (1 + (z / dist) ** 2)) * np.sqrt(
        ((z * x / dist ** 3) * np.sqrt(2) * pixel_size) ** 2 +
        ((z * y / dist ** 3) * np.sqrt(2) * pixel_size) ** 2 +
        (np.sqrt(2) * dz / dist) ** 2
    )

    uncertainties_list.append(round(delta_slope, 2))


def slopes_stopar_calculation(demi_profile_values, demi_profile_coords_relative, max_coords_real, max_values,
                              point_inner, idx_inner, crater_floor, pixel_size, dz, out_transform, no_data_value, zone):
    '''
    Compute crater slopes using the Stopar et al. (2017) method.

    This function computes slopes pixel-to-pixel within crater profiles between specified boundaries.
    It also calculates uncertainties and geometries related to these slopes.

    Parameters:
    -----------
    demi_profile_values: list of lists or arrays
        Elevation values of points on semi-profiles

    demi_profile_coords_relative: list of lists or arrays
        Relative coordinates of points on semi-profiles

    max_coords_real: list of tuples
        Real-world coordinates of maximum elevation points on crater rim

    max_values: list or array
        Elevations of the maximum points on the crater rim

    point_inner: list
        Inner boundary points for slope calculations

    idx_inner: list of tuples
        Indices defining the inner boundary range for slope calculation (start, end)

    crater_floor: list
        Indices of the crater floor points within the profiles

    pixel_size: int or float
        Size of one pixel on the terrain (meters)

    dz: float
        Vertical uncertainty (elevation measurement error)

    out_transform: affine transform
        Rasterio transform object to convert raster coordinates to real-world coordinates

    no_data_value: float or int
        Raster no-data value used to identify missing data

    zone: str or int
        Zone identifier for loading the appropriate raster files

    Returns:
    --------
    tuple:
        slopes_px_to_px: list
            List of computed slopes pixel-to-pixel (degrees)

        geom: list of shapely.geometry.LineString
            Geometries representing slope lines between floor and inner points

        mean_slope_px_to_px: float
            Mean of all pixel-to-pixel slopes (degrees)

        uncertainty_slope_px_to_px: list
            List of uncertainties associated with pixel-to-pixel slopes
    '''
    raster_path = f"../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation_crop.TIF"
    raster_fiability_path = f"../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation_fiabilite_crop.TIF"

    with rasterio.open(raster_path) as raster_pre_impact, rasterio.open(raster_fiability_path) as raster_fiability:
        elevation_pre_impact = []
        diff_pre_impact = []
        fiabilite = []


        slopes_px_to_px = []
        uncertainty_slope_px_to_px = []
        geom = []

        for i, (profil_coords, profil_values) in enumerate(zip(demi_profile_coords_relative, demi_profile_values)):
            coord = max_coords_real[i]
            elevation = list(raster_pre_impact.sample([coord]))[0][0]
            elevation_pre_impact.append(elevation)

            fiab = list(raster_fiability.sample([coord]))[0][0]
            fiabilite.append(fiab)

            m = len(profil_coords[0])
            demi_profil = [[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)]
            demi_profil = np.where(demi_profil == no_data_value, np.nan, demi_profil)

            diff_pre_impact.append(max_values[i] - elevation)

            floor = [profil_coords[0][crater_floor[i]], profil_coords[1][crater_floor[i]], profil_values[crater_floor[i]]]

            s = []
            s_uncertainties = []

            mean_slope_px, mean_uncertainty = slopes_px2px(
                s, s_uncertainties, crater_floor[i], idx_inner[i][1],
                demi_profil, pixel_size, dz
            )

            slopes_px_to_px.append(mean_slope_px)
            uncertainty_slope_px_to_px.append(mean_uncertainty)

            geom.append(LineString([
                rasterio.transform.xy(out_transform, floor[0], floor[1]),
                rasterio.transform.xy(out_transform, point_inner[i][1][0], point_inner[i][1][1])
            ]))

    return slopes_px_to_px, geom, round(np.mean(slopes_px_to_px), 2), uncertainty_slope_px_to_px, diff_pre_impact, \
           fiabilite
