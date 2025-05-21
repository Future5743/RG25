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


def slopes_calculation(demi_profils_value, demi_profils_coords_relatives, index_maximum, pixel_size, dz, out_transform,
                       no_data_value, rate):
    '''
    This function compute the crater's slopes and their uncertainties with a given percentage.
    This method simply compute the slope formed by the profile given some percentage (e.g. a percentage of 20% leads to
    the calculation of the slope between the point at 20% of the distance between the lowest and the highest point and
    the point at 80% of the distance between the lowest and the highest point)

    Entries:
        demi_profils_value: list                    -- Contains the elevation value of each point on the semi-profiles
        demi_profils_coords_relatives: list         -- Contains the relative coordinates of each point on the
                                                       semi-profiles
        index_maximum: list                         -- Contains the index of the maximum point of each semi-profile
        pixel_size: int                             -- Size of the pixel on the terrain
        dz: float                                   -- Vertical uncertainty
        out_transform: ???                          -- ???
        rate: float                                 -- The wanted percentage to compute the slope

    Exit data:
        slopes: list                                -- Contains the slopes value of each semi-profiles
        delta_slopes: list                          -- Contains the uncertainties of each slope value for every
                                                       semi-profiles
        mean_slopes: list                           -- Contains the mean slope of each semi-profile
        slope_geometry: list                        -- Contains the geometry of each slope for every semi-profiles

    '''

    slopes, delta_slopes, slope_geometry = [], [], []

    demi_profils_value_function = demi_profils_value
    demi_profils_coords_relatives_function = demi_profils_coords_relatives

    index_demi_profil = 0

    for demi_profil in demi_profils_coords_relatives_function:

        demi_profils_value_function[index_demi_profil] = np.where(demi_profils_value_function[index_demi_profil] == no_data_value,
                                                                  np.nan,
                                                                  demi_profils_value_function[index_demi_profil])

        demi_profil[0] = demi_profil[0][:index_maximum[index_demi_profil]]
        demi_profil[1] = demi_profil[1][:index_maximum[index_demi_profil]]

        demi_profils_value_function[index_demi_profil] = demi_profils_value_function[index_demi_profil][len(demi_profil):]

        pos_point_min = int(rate * len(demi_profil[0]))

        while np.isnan(demi_profils_value_function[index_demi_profil][pos_point_min]):
            pos_point_min += 1

        pos_point_max = int((1-rate) * len(demi_profil[1]))

        while np.isnan(demi_profils_value_function[index_demi_profil][pos_point_max]):
            pos_point_max -= 1

        point_min = [demi_profil[0][pos_point_min], demi_profil[1][pos_point_min]]

        point_max = [demi_profil[0][pos_point_max], demi_profil[1][pos_point_max]]

        dist = distance_calculation(point_min, point_max, pixel_size)

        delta_z = demi_profils_value_function[index_demi_profil][pos_point_max] - \
                  demi_profils_value_function[index_demi_profil][pos_point_min]

        slope_rad = np.arctan(delta_z / dist)
        slope_deg = np.rad2deg(slope_rad)

        real_coord_point_min = rasterio.transform.xy(out_transform, point_min[0], point_min[1])

        real_coord_point_max = rasterio.transform.xy(out_transform, point_max[0], point_max[1])

        slope_geometry.append(LineString([real_coord_point_min, real_coord_point_max]))

        x = point_min[0] - point_max[0]
        y = point_min[1] - point_max[1]
        z = delta_z

        delta_slope = (1 / (1 + (z / dist)**2)) * np.sqrt(
            ((z * x / dist**3) * np.sqrt(2) * pixel_size)**2 +
            ((z * y / dist**3) * np.sqrt(2) * pixel_size)**2 +
            (np.sqrt(2) * dz / dist)**2
        )

        slopes.append(round(slope_deg, 2))
        delta_slopes.append(round(delta_slope, 2))

        index_demi_profil += 1

    mean_slopes = round(np.nanmean(slopes), 2)

    return slopes, delta_slopes, mean_slopes, slope_geometry


def compute_slope(points):
    '''
    This function compute slopes in degrees with sa 3D PCA.

    Entries:
        points: array                                                         -- Points used to compute the PCA

    Exit data:
        abs(np.rad2deg(np.arctan(direction[2] / horizontal_norm))): float     -- Slope of the semi-profile
    '''

    mean = points.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(points - mean)
    direction = pca.components_[0]
    horizontal_norm = np.linalg.norm(direction[:2])
    if horizontal_norm == 0:
        return 0
    return abs(np.rad2deg(np.arctan(direction[2] / horizontal_norm)))


def monte_carlo_uncertainty(points, n_simulations, dx, dz):
    '''
    Compute the uncertainties on a slope with a Monte Carlo simulation

    Entries:
        points: array                          -- Points used to compute the Monte Carlo simulation
        n_simulations: int                     -- Number of wanted simulations
        dx: int                                -- Horizontal uncertainty (here the pixel-size)
        dz: float                              -- Vertical uncertainty

    Exit data:
        round(np.std(slopes), 2): float        -- Uncertainty of the slope calculation with the PCA method
    '''

    slopes = []
    for _ in range(n_simulations):
        noisy_points = points + np.stack([
            np.random.normal(0, dx, points.shape[0]),
            np.random.normal(0, dx, points.shape[0]),
            np.random.normal(0, dz, points.shape[0])
        ], axis=1)
        slopes.append(compute_slope(noisy_points))
    return round(np.std(slopes), 2)


def visualize_profile(points, slope, uncertainty, index):
    '''
    This function compute the 3D visualization of the emi-profile and the 3D result of the PCA.

    Entries:
        points: array          -- Points used to compute the PCA
        slope: float           -- Slope obtained by the PCA method
        uncertainty: float     -- Uncertainty of the slope calculation with the PCA method
        index: int             -- id of the semi-profile

    Exit data:
        no exit data
    '''

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Profil')

    mean = points.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(points - mean)
    direction = pca.components_[0]

    line_pts = np.array([
        mean - direction * 10,
        mean + direction * 10
    ])
    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color='red', linewidth=2, label='Direction ACP')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude')
    ax.set_title(f'Profil {index} - Pente: {slope:.2f}° ± {uncertainty:.2f}°')
    ax.legend()
    plt.tight_layout()
    plt.show()


def slope_calculation_by_PCA(demi_profils_value, demi_profils_coords_relatives, index_maximum,
                             out_transform, pixel_size, dz, n_simulations=100, visualize=False):
    '''
    This function compute the slopes of each semi-profils with a PCA and their uncertainties associated with a Monte
    Carlo simulation.

    Entries:
        demi_profils_value: list                           -- Contains th elevation value of each poin on the
                                                              semi-profiles
        demi_profils_coords_relatives: list                -- Contains the relative coordinates of each point on the
                                                              semi-profiles
        index_maximum: list                                -- Contains the index of the maximum point of each
                                                              semi-profiles
        out_transform: ???                                 -- ??
        pixel_size: int                                    -- Size of the pixel on the terrain
        dz: float                                          -- Vertical uncertainty
        n_simulations: int                                 -- Number of wanted simulations
        visualize: bool                                    -- True if the user want to visualize the semi_profile

    Exit data:
        slopes_PCA: list                                   -- Contains the slopes computed by PCA of each semi-profiles
        uncertainties: list                                -- Contains the uncertainties associates with the slopes of
                                                              each semi-profiles
        mean_slopes_PCA: float                             -- Is the mean of every slope of a crater
    '''
    slopes_PCA = []
    uncertainties = []

    for i in range(len(demi_profils_value)):
        # Nettoyage du profil
        demi_profils_value[i] = demi_profils_value[i][: index_maximum[i]]
        demi_profils_coords_relatives[i][0] = demi_profils_coords_relatives[i][0][: index_maximum[i]]
        demi_profils_coords_relatives[i][1] = demi_profils_coords_relatives[i][1][: index_maximum[i]]

        # Conversion coordonnées raster → monde réel
        points = []
        for j in range(len(demi_profils_value[i])):
            x, y = rasterio.transform.xy(out_transform,
                                         demi_profils_coords_relatives[i][0][j],
                                         demi_profils_coords_relatives[i][1][j])
            points.append([x, y, demi_profils_value[i][j]])
        points = np.array(points)
        points = points[~np.isnan(points).any(axis=1)]

        slope = compute_slope(points)
        uncertainty = monte_carlo_uncertainty(points, n_simulations, pixel_size, dz)

        slopes_PCA.append(round(slope, 2))
        uncertainties.append(uncertainty)

        if visualize:
            visualize_profile(points, slope, uncertainty, i)

        mean_slopes_PCA = round(np.mean(slopes_PCA), 2)

    return slopes_PCA, mean_slopes_PCA, uncertainties


def smooth_profile(demi_profile, window_size=5):
    '''
    This function smooth the demi-profile using a moving average while ignoring NaN values.

    Entries :
        demi_profile: list              -- Contains the elevation data of the given demi-profile
        window_size: int                -- The size of the window to smooth the profile

    Exit data:
        smoothed: list                  -- The elevation of the smoothed demi-profile
    '''
    demi_profile = np.array(demi_profile, dtype=np.float64)
    smoothed = np.full_like(demi_profile, np.nan)

    half_window = window_size // 2

    for i in range(len(demi_profile)):
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(demi_profile))
        window = demi_profile[start:end]
        smoothed[i] = np.nanmean(window) if np.isnan(window).sum() < len(window) else np.nan

    return smoothed


def compute_steepest_slope_3d(coords, elevations, index_max):
    '''
    This function computes the steepest uphill slope segment in a 3D profile, ignoring the crater's lowest point

    Entries:
        coords: list                -- Contains the coordinates of each point of the given profile
        elevations: list            -- Contains the elevation data of the given demi-profile
        min_distance: int           -- The minimum size of the window to compute the inner slope

    Exit data:
        max_slope: float            -- The  value in degrees  slope of the inner slope
        best_segment: list          -- The coordinates (long, lat, elevation) of the two points delimiting the inner
                                       slope
    '''
    coords = np.array(coords)[1:index_max]
    elevations = np.array(elevations)[1:index_max]

    n = len(coords)
    if n <= 1:
        return None, None

    min_win = min(int(0.5 * n), n - 1)
    max_win = min(int(0.9 * n), n - 1)

    max_slope = -np.inf
    best_segment = None

    for win in range(min_win, max_win + 1):
        for i in range(n - win):
            z1, z2 = elevations[i], elevations[i + win]
            if np.isnan(z1) or np.isnan(z2):
                continue

            p1, p2 = coords[i], coords[i + win]
            horiz_dist = np.linalg.norm(p2 - p1)
            if horiz_dist == 0:
                continue

            vert_dist = z2 - z1
            slope_deg = np.degrees(np.arctan2(vert_dist, horiz_dist))

            if slope_deg > max_slope:
                max_slope = slope_deg
                best_segment = [(p1[0], p1[1], z1), (p2[0], p2[1], z2)]

    return max_slope, best_segment


def save_smoothed_profile(smoothed, coords, index, zone, crater_id, swirl_status, index_max):
    '''
    This functiion save a 3D plot of the smoothed demi-profile with its steepest slope segment.

    Entries:
        smoothed: list                  -- The elevation of the smoothed demi-profile
        coords: list                    -- Contains the reel coordinates of each demi-profiles
        index: int                      -- Index of the studied demi-profile
        zone: str                       -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)
        crater_id: int                  -- ID of the studied crater
        swirl_status: str               -- Indicate if the crater is on or off swirl

    Exit data:
        No exit data
    '''
    coords = np.array(coords)
    smoothed = np.array(smoothed)

    slope_deg, segment = compute_steepest_slope_3d(coords, smoothed, index_max)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coords[:, 0], coords[:, 1], smoothed, marker='x', color='orange', label='Profil lissé')

    if segment:
        seg = np.array(segment)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='blue', linewidth=4, label=f'Slope ({slope_deg:.1f}°)')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")
    ax.set_title(f'Demi-profil {index * 10}°')
    ax.legend()

    subfolder = 'on_swirl' if swirl_status == 'on-swirl' else 'off_swirl'
    path = os.path.join('results', f'RG{zone}', 'profils', subfolder, str(crater_id), 'inner_slope')
    os.makedirs(path, exist_ok=True)

    filename = f'Profil3D_demi_{index * 10}.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()


### Monte Cralo

def estimate_uncertainty_per_profile(demi_profils_value, demi_profils_coords_relatives, index_maximum, out_transform,
                                     dx, dy, n_iter=500):
    slope_uncertainties = []

    for i, (values, (rows, cols)) in enumerate(zip(demi_profils_value, demi_profils_coords_relatives)):
        slopes = []

        for _ in range(n_iter):
            # Perturber les coordonnées
            perturbed_coords = []
            for r, c in zip(rows, cols):
                x, y = rasterio.transform.xy(out_transform, r, c)
                x += np.random.normal(0, dx)
                y += np.random.normal(0, dx)
                perturbed_coords.append([x, y])

            # Perturber les altitudes
            perturbed_values = np.array(values) + np.random.normal(0, dy, size=len(values))

            # Lissage et pente
            smoothed = smooth_profile(perturbed_values, window_size=5)
            slope_deg, _ = compute_steepest_slope_3d(perturbed_coords, smoothed, index_maximum[i])

            slopes.append(slope_deg)

        # Nettoyer les NaN éventuels
        slopes = np.array(slopes)
        slopes = slopes[~np.isnan(slopes)]

        if len(slopes) == 0:
            slope_uncertainties.append(np.nan)
        else:
            slope_uncertainties.append(round(np.std(slopes), 2))

    return slope_uncertainties




def inner_slopes(inner_slopes_deg, inner_slopes_delimitation, demi_profils_value, demi_profils_coords_relatives, zone,
                 crater_id, swirl_on_or_off, out_transform, index_maximum, dx, dy):
    '''
    This function compute a crater's inner slopes, and save it.

    Entries:
        inner_slopes_deg: list                      -- Contains nothing. Is filled within the function
                                                       Is meant to contain the value of the slope of each demi_profiles
        inner_slopes_delimitation: list             -- Contains nothing. Is filled within the function
                                                       Is meant to contain the delimitation of the inner slope of each
                                                       demi-profile
        demi_profils_value:list                     -- Contains the elevation value of each demi-profiles
        demi_profils_coords_relatives: list         -- Contains the relative coordinates of each demi-profiles
        zone: str                                   -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or
                                                       7)
        crater_id: int                              -- ID of the studied crater
        swirl_on_or_off: str                        -- Indicate if the crater is on or off swirl
        out_transform: ???                          -- ???

    Exit data:
        mean_carter_slope: float                    -- Is the mean of all the slopes of a crater
        slope_uncertainty: list                     -- Is the uncertainty of the
    '''
    for i, (values, (rows, cols)) in enumerate(zip(demi_profils_value, demi_profils_coords_relatives)):
        demi_coords = [list(rasterio.transform.xy(out_transform, r, c)) for r, c in zip(rows, cols)]

        smoothed = smooth_profile(values, window_size=5)
        slope_deg, segment = compute_steepest_slope_3d(demi_coords, smoothed, index_maximum[i])

        inner_slopes_deg.append(slope_deg)
        inner_slopes_delimitation.append(segment)

        save_smoothed_profile(smoothed, demi_coords, i, zone, crater_id, swirl_on_or_off, index_maximum[i])

    mean_crater_slope = np.mean(inner_slopes_deg)
    # slope_uncertainty = estimate_uncertainty_per_profile(demi_profils_value, demi_profils_coords_relatives,
    #                                                      index_maximum, out_transform, dx=5, dy=2, n_iter=500)

    return round(mean_crater_slope, 2)


'''
# Profil Horizontal
    # Trouver la plus basse altitude et la plus haute altitude entre celle de l'Est 
    # et celle de l'Ouest
    petite_altitude = min(max_val_right, max_val_left)
    grande_altitude = max(max_val_right, max_val_left)

    # Calculer la différence d'altitude
    altitude_difference = round(grande_altitude - petite_altitude, 4)

    # Calculer la pente en radians avec NumPy
    slope_radians = round(np.arctan(altitude_difference / distance_right_left), 4)

    # Convertir la pente en degrés
    slope_degrees_eo = round(np.degrees(slope_radians), 4)

# Profil Vertical
    # Trouver la plus basse altitude et la plus haute altitude entre celle du Nord 
    # et celle du Sud
    petite_altitude_ns = min(max_val_top, max_val_bas)
    grande_altitude_ns = max(max_val_top, max_val_bas)

    # Calculer la différence d'altitude
    altitude_difference_ns = round(grande_altitude_ns - petite_altitude_ns, 4)

    # Calculer la pente en radians avec NumPy
    slope_radians_ns = round(np.arctan(altitude_difference_ns / distance_bas_top), 4)

    # Convertir la pente en degrés
    slope_degrees_ns = round(np.degrees(slope_radians_ns), 4)

# Pente max entre les deux profils du cratère
    pente_max = round(max(slope_degrees_ns, slope_degrees_eo), 1)
'''

def slopes_stopar_calculation(demi_profils_value, demi_profils_coords_relatives, pixel_size, out_transform,
                              no_data_value, rate):
    slopes_px_to_px = []
    slopes = []
    geom = []

    for i, (profil_coords, profil_values) in enumerate(zip(demi_profils_coords_relatives, demi_profils_value)):
        m = len(profil_coords[0])
        demi_profil = [[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)]

        demi_profil = np.where(demi_profil == no_data_value, np.nan, demi_profil)

        # Trouver le point maximum (fin) non NaN
        i_max = -1
        while np.isnan(demi_profil[i_max][2]):
            i_max -= 1
        point_max = demi_profil[i_max][2]

        # Trouver le point minimum (début) non NaN
        i_min = 0
        while np.isnan(demi_profil[i_min][2]):
            i_min += 1
        point_min = demi_profil[i_min][2]

        depth_total = point_max - point_min
        alt_min = rate * depth_total + point_min
        alt_max = (1-rate) * depth_total + point_min

        # Initialisation des points internes les plus proches des altitudes cibles
        point_inner_min = point_inner_max = None
        index_min_inner = index_max_inner = -1
        min_dist_min = min_dist_max = np.inf

        for j, (_, _, z) in enumerate(demi_profil):
            if np.isnan(z):
                continue
            if abs(z - alt_min) < min_dist_min:
                min_dist_min = abs(z - alt_min)
                point_inner_min = demi_profil[j]
                index_min_inner = j
            if abs(z - alt_max) < min_dist_max:
                min_dist_max = abs(z - alt_max)
                point_inner_max = demi_profil[j]
                index_max_inner = j

        # S'assurer de l'ordre des indices
        if index_min_inner > index_max_inner:
            index_min_inner, index_max_inner = index_max_inner, index_min_inner
            point_inner_min, point_inner_max = point_inner_max, point_inner_min

        if point_inner_min is None or point_inner_max is None:
            print(demi_profil)

            print(f"Profil {i}: point_inner_min ou point_inner_max est None")
            slopes_px_to_px.append(np.nan)
            slopes.append(np.nan)
            geom.append(None)
            continue

        # Calcul des pentes px à px
        s = []
        for j in range(index_min_inner, index_max_inner):
            pt1 = demi_profil[j]
            pt2 = demi_profil[j + 1]
            dist = distance_calculation(pt1[:-1], pt2[:-1], pixel_size)
            if dist == 0:
                continue
            dz = pt2[-1] - pt1[-1]
            s.append(round(np.rad2deg(np.arctan(dz / dist)), 2))

        mean_slope_px = round(np.nanmean(s), 2) if s else np.nan
        slopes_px_to_px.append(mean_slope_px)

        # Calcul de la pente entre les deux points internes
        dist_total = distance_calculation(point_inner_min[:-1], point_inner_max[:-1], pixel_size)

        if dist_total == 0:
            print(f"Profil {i}: Distance totale = 0 entre {point_inner_min[:-1]} et {point_inner_max[:-1]}")
            slopes.append(np.nan)
        else:
            depth = abs(point_inner_max[-1] - point_inner_min[-1])
            slope = round(np.rad2deg(np.arctan(depth / dist_total)), 2)
            slopes.append(slope)

        # Construction de la géométrie
        geom.append(LineString([
            rasterio.transform.xy(out_transform, point_inner_min[0], point_inner_min[1]),
            rasterio.transform.xy(out_transform, point_inner_max[0], point_inner_max[1])
        ]))

    return slopes, slopes_px_to_px, geom, round(np.mean(slopes), 2), round(np.mean(slopes_px_to_px), 2)






