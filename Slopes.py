########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import numpy as np
from sklearn.decomposition import PCA
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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



def slopes_calculation(min_pos, min_value, max_value, max_coord_relative, pixel_size, dz):
    '''
    Compute the crater's slopes and their uncertainties with a simple method.
    This method simply compute the slope formed by the highest and the lowest point on a semi-profile.

    Entries:
        min_pos: tupple                        -- Relative coordinates of the crater's lowest point
        min_value: float                       -- Elevation of the crater's lowest point
        max_value: list                        -- Contains all the elevations of the maxima on the rim
        max_coord_relative: list               -- Contains all the relative coordinates of the maxima on the rim
        pixel_size: int                        -- Size of the pixel on the terrain
        dz: float                              -- Vertical uncertainty


    Exit data:
        slopes: list                           -- Contains the slopes of each semi-profiles
        delta_slopes: list                     -- Contains the uncertainties associates with the slopes of each
                                                  semi-profiles
    '''

    min_pos = list(min_pos)
    if 0 in min_pos:
        min_pos.remove(0)

    slopes, delta_slopes = [], []
    for point in range(len(max_value)):
        dist = distance_calculation(min_pos, max_coord_relative[point], pixel_size)
        delta_z = max_value[point] - min_value
        slope_rad = np.arctan(delta_z / dist)
        slope_deg = np.rad2deg(slope_rad)

        x = min_pos[0] - max_coord_relative[point][0]
        y = min_pos[1] - max_coord_relative[point][1]  # ← Correction ici
        z = min_value - max_value[point]

        delta_slope = (1 / (1 + (z / dist)**2)) * np.sqrt(
            ((z * x / dist**3) * np.sqrt(2) * pixel_size)**2 +
            ((z * y / dist**3) * np.sqrt(2) * pixel_size)**2 +
            (np.sqrt(2) * dz / dist)**2
        )

        slopes.append(round(slope_deg, 2))
        delta_slopes.append(round(delta_slope, 2))

    return slopes, delta_slopes



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

    Exit data :
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

    Exit data :
        slopes_PCA: list                                   -- Contains the slopes computed by PCA of each semi-profiles
        uncertainties: list                                -- Contains the uncertainties associates with the slopes of
                                                              each semi-profiles
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

    return slopes_PCA, uncertainties




