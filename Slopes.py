######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import numpy as np
from sklearn.decomposition import PCA
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################
def distance_calculation(pos1, pos2, pixel_size_tb=2):
    pixel_dist_tb = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_in_meters_tb = pixel_dist_tb * pixel_size_tb

    return distance_in_meters_tb

def max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb):
    slopes = []
    for i in range(int(len(max_value) / 2)):
        # Calcul de la distance entre les points des crêtes opposées
        dist = distance_calculation(max_coord_relative[i], max_coord_relative[i + 18], pixel_size_tb)

        # Trouver la plus basse et la plus haute altitude pour les profils tous les 10 m
        low_alt = min(max_value[i], max_value[i + 18])
        high_alt = max(max_value[i], max_value[i + 18])

        # Calcul de la difference d'altitudes
        diff_alt = round(high_alt - low_alt, 4)

        # Calcul de la pente
        slope_rad = round(np.arctan(diff_alt / dist), 4)

        # Convertion en degres
        slope_deg = round(np.rad2deg(slope_rad), 4)

        slopes.append(slope_deg)

    return np.max(slopes)



def slopes_calculation(min_pos, min_value, max_value, max_coord_relative, pixel_size_tb, precision_error):
    min_pos = list(min_pos)
    min_pos.remove(0)
    slopes = []
    delta_slopes = []

    for point in range(len(max_value)):
        dist = distance_calculation(min_pos, max_coord_relative[point], pixel_size_tb)

        diff_alt = max_value[point] - min_value

        slope_rad = np.arctan(diff_alt / dist)



        slope_deg = round(np.rad2deg(slope_rad), 4)

        if slope_deg >50 :
            print(min_pos, max_coord_relative[point])
            print(slope_rad)
            print(diff_alt / dist)
            print(slope_deg, max_value[point], min_value, dist, diff_alt)

        slopes.append(slope_deg)

        x = min_pos[0] - max_coord_relative[point][0]
        y = min_pos[1] - max_coord_relative[point][0]
        z = min_value - max_value[point]


        delta_slope = (1 / (1 + (z / dist)**2) ) * np.sqrt(
            ((z * x / dist**3) * np.sqrt(2) * pixel_size_tb)**2
            + ((z * y / dist**3) * np.sqrt(2) * pixel_size_tb)
            + (np.sqrt(2) * precision_error / dist)**2
        )

        delta_slopes.append(delta_slope)

    return slopes, delta_slopes


def slope_calculation_by_PCA(profils, demi_profils_coords_relatives, index_maximum, out_transform, visualize=False):
    slopes_PCA = []

    for i in range(len(profils)):
        points = []

        profils[i] = profils[i][: index_maximum[i]]
        demi_profils_coords_relatives[i][0] = demi_profils_coords_relatives[i][0][: index_maximum[i]]
        demi_profils_coords_relatives[i][1] = demi_profils_coords_relatives[i][1][: index_maximum[i]]

        for j in range(len(profils[i])):
            x, y = rasterio.transform.xy(
                out_transform,
                demi_profils_coords_relatives[i][0][j],
                demi_profils_coords_relatives[i][1][j]
            )
            points.append([x, y, profils[i][j]])

        points = np.array(points)
        points = points[~np.isnan(points).any(axis=1)]

        # Centre des points
        mean_point = points.mean(axis=0)

        # ACP
        pca = PCA(n_components=1)
        pca.fit(points - mean_point)

        direction = pca.components_[0]

        # Calcul de la pente par projection 2D (Z / sqrt(X² + Y²))
        horizontal_norm = np.linalg.norm(direction[:2])
        if horizontal_norm == 0:
            pente_deg = 0
        else:
            pente_proj = direction[2] / horizontal_norm
            pente_deg = np.rad2deg(np.arctan(pente_proj))

        pente_deg = abs(pente_deg)  # toujours positive

        slopes_PCA.append(pente_deg)

        if pente_deg > 20:
            visualize = True
            print(visualize)

        # Visualisation du profil + droite ACP
        if visualize:  # juste pour le premier profil
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Profil')

            # Droite ACP
            line_length = 10
            line_pts = np.array([
                mean_point - direction * line_length,
                mean_point + direction * line_length
            ])
            ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color='red', label='Droite ACP', linewidth=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Altitude')
            ax.set_title(f'Profil {i} avec direction principale (Pente: {pente_deg:.2f}°)')
            ax.legend()
            plt.tight_layout()
            plt.show()

    return slopes_PCA





