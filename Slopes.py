########################################################################################################################
##################################################### IMPORTATIONS #####################################################
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
    This function compute the real distance between two points in meters.

    Entries:
        pos1: list, array or tupple         -- Relative coordinates of the first point
        pos2: list, array or tupple         -- Relative coordinates of the second point
        pixel_size: int                     -- Size of the pixel on the terrain

    Exit data:
        pixel_dist * pixel_size: float      -- Distance between the two points
    '''

    pixel_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return pixel_dist * pixel_size


def max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size):
    '''
    This function compute the maximum slope between two opposite sides of the crater

    Entries:
        max_value: list                        -- Contains all the elevations of the maxima on the rim
        max_coord_relative: list               -- Contains all the relative coordinates of the maxima on the rim
        pixel_size: int                        -- Size of the pixel on the terrain

    Exit data:
        np.max(slopes): float                  -- Maximum slope between two opposite sides of the crater
    '''

    slopes = []
    for i in range(len(max_value) // 2):
        dist = distance_calculation(max_coord_relative[i], max_coord_relative[i + 18], pixel_size)
        delta_z = abs(max_value[i] - max_value[i + 18])
        slope_deg = np.rad2deg(np.arctan(delta_z / dist))
        slopes.append(round(slope_deg, 4))
    return np.max(slopes)


def slopes_px2px(s, s_uncertainties, index_min_inner, index_max_inner, demi_profil, pixel_size, dz):
    '''
    This function compute the slope following the method used in Stopar et al., 2017.
    This method consist in compute the slope using all the points between rate and (1-rate) of the depth of the crater.
    Then the depth is computed by averaging the slope between each adjacent points.

    Entries:
        s: list                             -- Contains the slopes computed (the slopes between adjacent points)
        s_uncertainties: list               -- Contains the uncertainties associated with each slope computed (the
                                               slopes between adjacent points)
        index_min_inner: int                -- Index of the lowest point defining the boundary of the entire slope
        index_max_inner: int                -- Index of the highest point defining the boundary of the entire slope
        demi_profil: list                   -- Contains all the points defining the semi-profile
        pixel_size: int                     -- Size of the pixel on the terrain
        dz: float                           -- Vertical uncertainty

    Exit data:
        mean_slope_px: float                -- The average slope computed with all the slopes between adjacent points
        mean_uncertainty: float             -- The average uncertainty associated with each slope computed between
                                               adjacent points
    '''

    for j in range(index_min_inner, index_max_inner):
        pt1 = demi_profil[j]
        pt2 = demi_profil[j + 1]

        dist = distance_calculation(pt1[:-1], pt2[:-1], pixel_size)
        if dist == 0:
            continue

        depth = pt2[-1] - pt1[-1]

        slope = round(np.rad2deg(np.arctan(depth / dist)), 2)
        s.append(slope)

        # Suppose dz and pixel_size_uncertainty are known constants
        slope_uncertainties(s_uncertainties, pt1, pt2, dist, pixel_size, dz)

    if s:
        mean_slope_px = round(np.nanmean(s), 2)
        mean_uncertainty = round(np.sqrt(np.nansum(np.array(s_uncertainties) ** 2)) / len(s_uncertainties), 2)

    else:
        mean_slope_px = np.nan
        mean_uncertainty = np.nan

    return mean_slope_px, mean_uncertainty


def slope_uncertainties(uncertainty_slope, point_1, point_2, dist, pixel_size, dz):
    '''
    This function compute the uncertainties associated with the slope calculation.

    Entries:
        uncertainty_slope: list             -- Contains the uncertainties associated with the slopes calculation
        point_1: list                       -- Contains the coordinates and the elevation of the first studied point
                                               (one of the two delimitations of the slope computed)
        point_2: list                       -- Contains the coordinates and the elevation of the second studied point
                                               (one of the two delimitations of the slope computed)
        dist: float                         -- Distance between the two studied points
        pixel_size: int                     -- Size of the pixel on the terrain
        dz: float                           -- Vertical uncertainty

    Exit data:
        No exit data
    '''
    x = point_1[0] - point_2[0]
    y = point_1[1] - point_2[1]
    z = point_1[2] - point_2[2]

    delta_slope = (1 / (1 + (z / dist) ** 2)) * np.sqrt(
        ((z * x / dist ** 3) * np.sqrt(2) * pixel_size) ** 2 +
        ((z * y / dist ** 3) * np.sqrt(2) * pixel_size) ** 2 +
        (np.sqrt(2) * dz / dist) ** 2
    )

    uncertainty_slope.append(round(delta_slope, 2))


def point_profile_near_preimpact_surface(demi_profils_value, demi_profils_coords_relatives, elevation_preimpact):

    point_pre_impact = None
    idx_preimpact = 0
    dist_preimpact = np.inf

    for i in range(len(demi_profils_value)):
        if np.isnan(demi_profils_value[i]):
            continue
        if abs(elevation_preimpact - demi_profils_value[i]) < dist_preimpact:
            dist_preimpact, point_pre_impact, idx_preimpact = abs(elevation_preimpact - demi_profils_value[i]), \
                                                              [demi_profils_coords_relatives[0][i],
                                                               demi_profils_coords_relatives[1][i],
                                                               demi_profils_value[i]],\
                                                              i

    return point_pre_impact, idx_preimpact

def slopes_stopar_calculation(demi_profils_value, demi_profils_coords_relatives, max_coord_real, max_value, point_inner, idx_inner, crater_floor, pixel_size, dz,
                              out_transform, no_data_value, zone):
    '''
    This function compute the crater's slopes with the method used in Stopar et al., 2017.

    To be more precise, the method used in Stopar et al., 2017 is returned to the slopes_px_to_px variable.
    This method consist in compute the slope using all the points between rate and (1-rate) of the depth of the crater.
    Then the depth is computed by averaging the slope between each adjacent points.

    This function compute also an adaptation of the method used in Stopar et al., 2017.
    This methode only compute the slope with the point between rate and (1-rate) of the depth of the crater

    Entries:
        demi_profils_value: list                            -- Contains the elevation value of each point on the
                                                               semi-profiles
        demi_profils_coords_relatives: list                 -- Contains the relative coordinates of each point on the
                                                               semi-profiles
        pixel_size: int                                     -- Size of the pixel on the terrain
        dz: float                                           -- Vertical uncertainty
        out_transform: ???                                  -- ???
        no_data_value: ???                                  -- ???
        rate: float                                         -- The wanted percentage to compute the slope

    Exit data:
        slopes: list                                        -- Contains all the crater's slopes compute with an
                                                               adaptation of the method used in Stopar et al., 2017
        slopes_px_to_px: list                               -- Contains all the crater's slopes compute with the method
                                                               used in Stopar et al., 2017
        geom: list                                          -- Contains the geometry of all the slopes
        round(np.mean(slopes), 2): float                    -- Mean of all the crater's slopes compute with the
                                                               adaptation of the Stopar et al., 2017 method
        round(np.mean(slopes_px_to_px), 2): float           -- Mean of all the crater's slopes compute pixel to pixel
        uncertainty_slope: list                             -- Contains the uncertainties of all the crater's slopes
                                                               compute with an adaptation of the method used in Stopar
                                                               et al., 2017
        uncertainty_slope_px_to_px: list                    -- Contains the uncertainties of all the crater's slopes
                                                               compute with the method used in Stopar et al., 2017

    '''
    raster_path = f"../data/RG/DTM_interpolate/RG{zone}_interpolation_IDW_02_crop.TIF"
    raster_fiabilite_path = f"../data/RG/DTM_interpolate/RG{zone}_interpolation_IDW_02_crop_fiabilite.TIF"

    with rasterio.open(raster_path) as raster_pre_impact, rasterio.open(raster_fiabilite_path) as raster_fiabilite:
        slopes_px_to_px = []

        # Uncertainties
        uncertainty_slope_px_to_px = []

        geom = []
        elevation_pre_impact = []
        diff_pre_impact = []
        fiabilite = []

        # Étape 1 : prétraitement pour décider la logique globale
        for i, (profil_coords, profil_values) in enumerate(zip(demi_profils_coords_relatives, demi_profils_value)):
            coord = max_coord_real[i]
            elevation = list(raster_pre_impact.sample([coord]))[0][0]
            elevation_pre_impact.append(elevation)

            point_preimpact, index_preimpact = point_profile_near_preimpact_surface(
                demi_profils_value[i], demi_profils_coords_relatives[i], elevation)

            fiab = list(raster_fiabilite.sample([coord]))[0][0]
            fiabilite.append(fiab)

            m = len(profil_coords[0])
            demi_profil = [[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)]
            demi_profil = np.where(demi_profil == no_data_value, np.nan, demi_profil)

            profil_values_clean = np.array(profil_values)
            profil_values_clean = profil_values_clean[~np.isnan(profil_values_clean)]

            elevation = elevation_pre_impact[i]
            fiab = fiabilite[i]

            point_preimpact, index_preimpact = point_profile_near_preimpact_surface(
                demi_profils_value[i], demi_profils_coords_relatives[i], elevation)

            diff_pre_impact.append(max_value[i] - elevation)

            floor = [profil_coords[0][crater_floor[i]], profil_coords[1][crater_floor[i]],
                     profil_values[crater_floor[i]]]

            s = []
            s_uncertainties = []

            mean_slope_px, mean_uncertainty = slopes_px2px(
                s, s_uncertainties, crater_floor[i], idx_inner[i][1],
                demi_profil, pixel_size, dz
            )

            if mean_slope_px == 0:
                print(f"Fiabilité: {fiab}")
                print(f"point preimpact: {point_preimpact}")
                print(f"Point au sol: {floor}")

            slopes_px_to_px.append(mean_slope_px)
            uncertainty_slope_px_to_px.append(mean_uncertainty)

            geom.append(LineString([
                rasterio.transform.xy(out_transform, floor[0], floor[1]),
                rasterio.transform.xy(out_transform, point_inner[i][1][0], point_inner[i][1][1])
            ]))

    return slopes_px_to_px, geom, round(np.mean(slopes_px_to_px), 2), uncertainty_slope_px_to_px, diff_pre_impact, fiabilite


