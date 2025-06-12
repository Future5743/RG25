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


def slope_calculation_by_PCA(demi_profils_value, demi_profils_coords_relatives, index_maximum, out_transform,
                             pixel_size, dz, n_simulations=100, visualize=False):
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

    return round(mean_crater_slope, 2) #, slope_uncertainty


def slopes_point_inner_max_inner_min(slopes, uncertainty_slope, i, point_inner_min, point_inner_max, pixel_size, dz):
    '''
    This function compute an adaptation of the method used in Stopar et al., 2017.
    This methode only compute the slope with the point between rate and (1-rate) of the depth of the crater

    Entries:
        slopes: list                            -- Contains all the slopes computed following this method
        uncertainty_slope: list                 -- Contains all the uncertainties associated with each slope
        i: int                                  -- Index of the loop
        pixel_size: int                         -- Size of the pixel on the terrain
        dz: float                               -- Vertical uncertainty

    Exit data:
        No exit data
    '''

    # Calcul de la pente entre les deux points internes
    dist_total = distance_calculation(point_inner_min[:-1], point_inner_max[:-1], pixel_size)

    if dist_total == 0:
        print(f"Profil {i}: Distance totale = 0 entre {point_inner_min[:-1]} et {point_inner_max[:-1]}")
        slopes.append(np.nan)
    else:
        depth = abs(point_inner_max[-1] - point_inner_min[-1])
        slope = round(np.rad2deg(np.arctan(depth / dist_total)), 2)
        slopes.append(slope)

    slope_uncertainties(uncertainty_slope, point_inner_min, point_inner_max, dist_total, pixel_size, dz)



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

    raster_path = f"../data/RG/RG{zone}_interpolation_robuste_02.TIF"
    raster_fiabilite_path = f"../data/RG/RG{zone}_interpolation_robuste_02_fiabilite_crop.TIF"

    with rasterio.open(raster_path) as raster_pre_impact, rasterio.open(raster_fiabilite_path) as raster_fiabilite:
        print(raster_pre_impact)
        slopes = []
        slopes_px_to_px = []

        # Uncertainties
        uncertainty_slope = []
        uncertainty_slope_px_to_px = []

        geom = []
        elevation_pre_impact = []
        diff_pre_impact = []
        fiabilite = []

        for i, (profil_coords, profil_values) in enumerate(zip(demi_profils_coords_relatives, demi_profils_value)):
            m = len(profil_coords[0])
            demi_profil = [[profil_coords[0][j], profil_coords[1][j], profil_values[j]] for j in range(m)]
            demi_profil = np.where(demi_profil == no_data_value, np.nan, demi_profil)

            profil_values_clean = np.array(profil_values)
            profil_values_clean = profil_values_clean[~np.isnan(profil_values_clean)]

            coord = max_coord_real[i]  # (x, y) réel
            elevation = list(raster_pre_impact.sample([coord]))[0][0]  # lecture unique
            elevation_pre_impact.append(elevation)

            point_preimpact, index_preimpact = point_profile_near_preimpact_surface(demi_profils_value[i],
                                                                                   demi_profils_coords_relatives[i],
                                                                                   elevation)


            fiab = list(raster_fiabilite.sample([coord]))[0][0]
            fiabilite.append(fiab)

            diff_pre_impact.append(max_value[i] - elevation)

            if fiab > 0.8 and index_preimpact > len(demi_profils_value[i]) * 1/3:
                point_rim = point_preimpact
                idx_rim = index_preimpact
            else:
                point_rim = point_inner[i][1]
                idx_rim = idx_inner[i][1]

            floor = [profil_coords[0][crater_floor[i]], profil_coords[1][crater_floor[i]],
                     profil_values[crater_floor[i]]]

            slopes_point_inner_max_inner_min(slopes, uncertainty_slope, i, floor, point_rim,
                                             pixel_size,
                                             dz)

            s = []
            s_uncertainties = []

            mean_slope_px, mean_uncertainty = slopes_px2px(
                s, s_uncertainties, crater_floor[i], idx_rim,
                demi_profil, pixel_size, dz
            )

            if mean_slope_px == 0:
                print(f"Fiabilité: {fiab}")
                print(f"point preimpact: {point_preimpact}")
                print(f"POint au sol: {floor}")

            slopes_px_to_px.append(mean_slope_px)
            uncertainty_slope_px_to_px.append(mean_uncertainty)

            geom.append(LineString([
                rasterio.transform.xy(out_transform, floor[0], floor[1]),
                rasterio.transform.xy(out_transform, point_rim[0], point_rim[1])
            ]))

    return slopes, slopes_px_to_px, geom, round(np.mean(slopes), 2), round(np.mean(slopes_px_to_px), 2), \
           uncertainty_slope, uncertainty_slope_px_to_px

