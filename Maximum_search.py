########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import geopandas as gpd
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

    # Delete the 0 to only have real data
    max_values = [x for x in max_values if x != 0]
    max_coords_relative = [x for x in max_coords_relative if x != 0]
    max_coords_real = [x for x in max_coords_real if x != 0]
    max_geometries = [x for x in max_geometries if x != 0]

    return lowest_point_coord, min_geometry, not_enough_data, max_values, max_coords_relative, max_coords_real, \
           max_geometries, half_profiles_values, half_profiles_coords_relative
