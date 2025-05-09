# 11/04/2025. Code en cours de mise à jour.

########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import geopandas as gpd    # Import of the “Geopandas” python library. Allows you to manipulate geographic data
import os
import shutil
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from Maximum_search import Finding_maxima, horizontal_90, horizontal_270, vertical_360, vertical_180
from Circularity import Miller_index
from Slopes import max_crater_slopes_calculation, slopes_calculation, slope_calculation_by_PCA
from TRI import TRI
from Topographical_profiles import main

########################################################################################################################
##################################################### DATA OPENING #####################################################
########################################################################################################################

zone = '7'

# Définition des tailles de pixel et erreurs de précision par zone
zone_settings = {
    '1': {'pixel_size_tb': 5, 'precision_error': 5.1},
    '2': {'pixel_size_tb': 2, 'precision_error': 0.81},
    '3': {'pixel_size_tb': 2, 'precision_error': 0.91},
    '4': {'pixel_size_tb': 2, 'precision_error': 0.87},
    '5': {'pixel_size_tb': 5, 'precision_error': 2.54},
    '6': {'pixel_size_tb': 5, 'precision_error': 2.34}
}

# Valeurs par défaut si la zone n'est pas reconnue
default_settings = {'pixel_size_tb': 5, 'precision_error': 2.37}

# Récupère les paramètres selon la zone
params = zone_settings.get(zone, default_settings)
pixel_size_tb = params['pixel_size_tb']
precision_error = params['precision_error']

# Chemins vers les fichiers de données
crater_shapefile_path = os.path.join('data', 'Buffer_crateres', f'Buffer_RG{zone}')
raster_path = os.path.join('..', 'data', 'DTM', f'NAC_DTM_REINER{zone}.tiff')
hiesinger_path = os.path.join('data', 'HIESINGER2011_MARE_AGE_UNITS_180', 'HIESINGER2011_MARE_AGE_UNITS_180.SHP')
swirls_path = os.path.join('data', 'Swirl', 'REINER_GAMMA.shp')

# Chargement des fichiers shapefiles en GeoDataFrame
craters = gpd.read_file(crater_shapefile_path)
hiesinger = gpd.read_file(hiesinger_path)
swirls = gpd.read_file(swirls_path)

# Extraction des géométries et âges
swirls_geom = swirls.geometry
hiesinger_geom = hiesinger.geometry
hiesinger_age = hiesinger['Model_Age']

########################################################################################################################
##################################################### LIST CREATION ####################################################
########################################################################################################################

# POINT geometry
highest_points = []             # Stores the geometry of the highest points corresponding to the crater crest

lowest_points = []              # Stores the geometry of the lowest points

profil_90 = []                  # Stores the geometry of the highest points of the ridge every 90°

centers = []                    # Stores the geometry of the centers of each valid crater found by YOLOv5

# LINESTRING geometry

Lines_visualisation = []       # Stores line geometry for viewing each profile

# POLYGON geometry

rim_approx = []                 # Stores the geometry resulting from the polygon formed by the highest_points

rim_approx_smooth = []          # Stores the smoothed geometry resulting from the polygon formed by the highest_points

# List to store information about the slope between opposite edges of a crater
results_slopes = []

# List to store information about a crater's circularity for all selected craters
results_circularity = []

# Geometry of final craters
result_geom_select_crat = []

# List to store the information needed for the final calculation of the dD ratio of each crater
results_ratio_dD = []

# Force full display of a numpy array
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
if os.path.exists("results/RG" + zone + "/profils"):
    try:
        shutil.rmtree("results/RG" + zone + "/profils")
    except OSError as e:
        print(f"Error: {e.strerror}")


if os.path.exists("results/RG" + zone + "/TRI"):
    try:
        shutil.rmtree("results/RG" + zone + "/TRI")
    except OSError as e:
        print(f"Error: {e.strerror}")

# Open raster file
with rasterio.open(raster_path) as src:
    raster_array = src.read(1)
    no_data_value = src.nodata

    for _, crater in tqdm(craters.iterrows(), total=len(craters)):
        # Données du cratère
        crater_geom = [crater.geometry]
        crater_id = crater.run_ID
        nac_id = crater.NAC_DTM_ID
        center_x, center_y = crater.center_lon, crater.center_lat
        radius = crater.ray_maxdia
        coord_center = (center_x, center_y)

        coord_center_geom = Point(coord_center)

        # Découpe du raster avec le cratère
        try:
            out_image, out_transform = mask(src, crater_geom, crop=True)
        except ValueError:
            continue  # Skip invalid geometries or ones falling outside raster extent

        # Masquage des valeurs no_data
        masked_image = np.ma.masked_equal(out_image, no_data_value)

        # Création d'un GeoDataFrame pour le centre du cratère
        crater_center = gpd.GeoDataFrame(
            [{'geometry': Point(center_x, center_y)}],
            crs=craters.crs
        )

        ### --- HIESINGER --- ###
        crater_center = crater_center.to_crs(hiesinger.crs)

        # Vérifie si le centre est contenu dans un polygone Hiesinger
        matches_hiesinger = hiesinger.geometry.contains(crater_center.geometry.iloc[0])

        if not matches_hiesinger.any():
            continue  # Pas d'info d'âge pour ce cratère

        floor_age = hiesinger.loc[matches_hiesinger, 'Model_Age'].values[0]

        ### --- SWIRL --- ###
        crater_center = crater_center.to_crs(swirls.crs)
        matches_swirl = swirls.geometry.contains(crater_center.geometry.iloc[0])
        swirl_on_or_off = 'on-swirl' if matches_swirl.any() else 'off-swirl'

        ### --- LOW ELEVATION --- ###
        if masked_image.count() > 0:
            min_val = round(masked_image.min(), 4)
            min_pos = np.unravel_index(masked_image.argmin(), masked_image.shape)

        D = masked_image.shape[1] * 2

        ### --- HIGH ELEVATION --- ###

        # Initialize lists to store future data
        max_value = []                      # Stores altitude values for highest_points
        max_coord_relative = []             # Stores relative coordinates of highest_points
        max_coord_real = []                 # Stores the actual coordinates of the highest_points
        max_geom = []                       # Stores geometries of highest_points
        line_geom = []                      # Stores the line geometries of the profiles studied
        demi_profils_value = []             # Stores topographic profiles
        demi_profils_coords_relatives = []  # Stores the relative coordinates of points in the profile
        index_maximum = []                  # Stores the index of the maximum point of each semi-profiles

        lowest_point_coord, min_geom = Finding_maxima(min_pos, min_val, D, masked_image, out_transform, max_value,
                                                      max_coord_relative, max_coord_real, max_geom, line_geom,
                                                      demi_profils_value, demi_profils_coords_relatives,
                                                      index_maximum)

        if len(max_geom) == 36:

        ### ELEVATION HAUTE : Horizontal - 90° (vers la droite dans le plan en 2-Dimensions)

            max_val_right, point_haut_horiz_90 = horizontal_90(min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Horizontal - 270° (vers la gauche dans le plan en 2-Dimensions)

            max_val_left, point_haut_horiz_270 = horizontal_270(min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Vertical - 360° (vers le haut dans le plan en 2-Dimensions)

            max_val_top, point_haut_vert_360 = vertical_360 (min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Vertical - 180° (vers le bas dans le plan en 2-Dimensions)

            max_val_bas, point_haut_vert_180 = vertical_180(min_pos, masked_image, no_data_value, out_transform)

            ### --- DIAMETERS --- ###
            D = []

            # Calculation of different diameter values
            def calcul_distance(pos1, pos2, pixel_size_tb):

                pixel_dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                return pixel_dist * pixel_size_tb

            for i in range(int(len(max_coord_relative) / 2)):
                pos1 = max_coord_relative[i]
                pos2 = max_coord_relative[i + 18]

                d = calcul_distance(pos1, pos2, pixel_size_tb)
                D.append(d)

            D = np.array(D)

            # Average diameters
            moy_diam = round(np.mean(D), 2)

            # Uncertainity of a diameter
            delta_D_stopar = np.sqrt(2) * pixel_size_tb                                 # Stopar et al., 2017

            delta_D_hoover = np.sqrt(np.std(D)**2 + pixel_size_tb**2)                   # Hoover et al., 2024

            # Uncertainty of average
            N = len(D)
            delta_Dbarre_stopar = round(delta_D_stopar / np.sqrt(N), 2)                 # Stopar et al., 2017

            delta_Dbarre_hoover = round(delta_D_hoover / np.sqrt(N), 2)                 # Hoover et al., 2024


            # Calculation of the radius of the average diameter
            ray_largest_diam = round(moy_diam / 2, 1)

            if calcul_distance(lowest_point_coord, coord_center, pixel_size_tb) < moy_diam * 0.25 \
                    and moy_diam >= 40:

                '''
            # Profil vertical
                # Calcul de la distance euclidienne en pixels entre max_pos_bas et max_pos_top 
                # et la convertir en mètres
                def calculate_pixel_distance_tb(pos1tb, pos2tb, pixel_size_tb=5):

                    # Calculer la distance euclidienne en pixels
                    pixel_distance_tb = np.sqrt((pos2tb[0] - pos1tb[0]) ** 2 + (pos2tb[1] - pos1tb[1]) ** 2)

                    # Convertir la distance en mètres
                    distance_in_meters_tb = pixel_distance_tb * pixel_size_tb
                    return distance_in_meters_tb

                # Conversion de la distance euclidienne en mètres
                distance_bas_top = calculate_pixel_distance_tb(max_pos_bas, max_pos_top)

            # Profil horizontal
                # Calcul de la distance euclidienne en pixels entre max_pos_right et max_pos_left 
                # et la convertir en mètres
                def calculate_pixel_distance_lr(pos1lr, pos2lr, pixel_size_lr=5):

                    # Calculer la distance euclidienne en pixels
                    pixel_distance_lr = np.sqrt((pos2lr[0] - pos1lr[0]) ** 2 + (pos2lr[1] - pos1lr[1]) ** 2)

                    # Conversion de la distance euclidienne en mètres
                    distance_in_meters_lr = pixel_distance_lr * pixel_size_lr
                    return distance_in_meters_lr

                # Conversion de la distance euclidienne en mètres
                distance_right_left = calculate_pixel_distance_lr(max_pos_right, max_pos_left)
                '''

            # Calculation of Miller's circularity index
                circularity = Miller_index(min_pos, max_coord_relative, pixel_size_tb)
                circularity = round(circularity, 2)

                if 0.97 <= circularity <= 1:

                    ### --- SLOPES --- ###

                    max_slope_crater = max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb)

                    if max_slope_crater < 8:

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


                        ### --- CREATION OF A CIRCLE ADJUSTED TO CRATER DIMENSIONS --- ###

                        def buffer_diam_max(center_x, center_y, radius, num_points=40):
                            center = Point(center_x, center_y)
                            buffer_poly = center.buffer(radius, resolution=num_points)
                            return buffer_poly

                        buf_diam_max = buffer_diam_max(lowest_point_coord[0], lowest_point_coord[1],
                                                       ray_largest_diam)

                    # # DISTANCE MIN_VAL AVEC CHACUN DES POINTS HAUTS E,O,N,S
                    #     def pixel_distance_centre_haut(pos1ch, pos2ch, pixel_size_ch=2):
                    #
                    #         # Calculer la distance euclidienne en pixels
                    #         pixel_distance_ch = np.sqrt((pos2ch[0] - pos1ch[0]) ** 2
                        #         + (pos2ch[1] - pos1ch[1]) ** 2)
                    #
                    #         # Conversion de la distance euclidienne en mètres
                    #         distance_in_meters_ch = pixel_distance_ch * pixel_size_ch
                    #         return distance_in_meters_ch
                    #
                    #     # Conversion de la distance euclidienne en mètres
                    #     distance_centre_haut = pixel_distance_centre_haut(lowest_points[0], highest_points[0])
                    #     print(f"Distance entre le point haut au Nord et le point bas au centre :
                    #           {distance_centre_haut} mètres", crater_id)

                        ### --- AVERAGE CRATER DEPTH --- ###

                        depth = [x - min_val for x in max_value]

                        prof_moyen_crat = round(np.mean(depth), 3)

                        delta_d_stopar = np.sqrt(2) * precision_error / np.sqrt(N)      # Stopar et al., 2017

                        sigma = np.sqrt(precision_error ** 2 + np.std(depth) ** 2)
                        delta_d_hoover = sigma / np.sqrt(N + 1)                         # Hoover et al., 2024

                        ### --- d//D CALCULATION --- ###

                        ratio_dD = round(prof_moyen_crat / moy_diam, 3)

                        # d/D UNCERTAINTIES CALCULATION

                        ## Stopar et al., 2017
                        rel_err_prof_stopar = delta_d_stopar / prof_moyen_crat
                        rel_err_diam_stopar = delta_Dbarre_stopar / moy_diam
                        rel_err_ratio_stopar = np.sqrt(rel_err_prof_stopar ** 2 + rel_err_diam_stopar ** 2)
                        delta_dD_stopar = round(rel_err_ratio_stopar * ratio_dD, 3)

                        ## Hoover et al., 2024
                        rel_err_prof_hoover = delta_d_hoover / prof_moyen_crat
                        rel_err_diam_hoover = delta_D_hoover / moy_diam
                        rel_err_ratio_hoover = np.sqrt(rel_err_prof_hoover ** 2 + rel_err_diam_hoover ** 2)
                        delta_dD_hoover = round(rel_err_ratio_hoover * ratio_dD, 3)

                # Ajouter les informations de Pente à la liste results_slopes
                #         results_slopes.append({'run_ID': crater_id,
                #                               'Altitude_PHN': round(max_val_top, 3),
                #                               'Altitude_PHE': round(max_val_right, 3),
                #                               'Altitude_PHS': round(max_val_bas, 3),
                #                               'Altitude_PHO': round(max_val_left, 3),
                #                               'GPS_PHN_x': max_x_top,
                #                               'GPS_PHN_y': max_y_top,
                #                               'GPS_PHE_x': max_x_rg,
                #                               'GPS_PHE_y': max_y_rg,
                #                               'GPS_PHS_x': max_x_bas,
                #                               'GPS_PHS_y': max_y_bas,
                #                               'GPS_PHO_x': max_x_lf,
                #                               'GPS_PHO_y': max_y_lf,
                #                               'SCR': src.crs.to_string(),
                #                               'Hauteur_PHS_PHN': round(altitude_difference_ns, 3),
                #                               'degres_pente_PHS_PHN': round(slope_degrees_ns, 2),
                #                               'Hauteur_PHE_PHO': round(altitude_difference, 3),
                #                               'degres_pente_PHE_PHO': round(slope_degrees_eo, 2)
                #                                })

                    # Ajouter les informations de Circularité à la liste results_circularity
                    #     results_circularity.append({'run_ID': crater_id,
                    #                                 'Diametre_PHS_PHN': distance_bas_top,
                    #                                 'Diametre_PHE_PHO': distance_right_left,
                    #                                 'Moyenne_diametre': moy_diam,
                    #                                 'Circularite': round(circularity, 2)
                    #                                 })

                        # Teste si le pixel haut Est, Ouest, Nord et Sud ne se trouve pas trop prêt de la limite du
                        # fichier de forme de polygone "selection_crateres.shp"

    ### Add geometry from highest_points
                        rim_approx_geom = Polygon(max_coord_real)

    ### Smoothing the polygon from the previous polygon

                        '''
                        def bezier_curve(p0, p1, p2, n=20) :
                            t = np.linspace(0,1,n)

                            return (1-t)[:, None]**2 * p0 + 2 * (1-t)[:, None] * t[:, None] * p1 
                                    + t[:, None]**2 * p2

                        def smooth_polygon_with_bezier (poly: Polygon, n=20) :
                            if not poly.is_valid:
                                raise ValueError("Polygone non valide")
                            coords = list(poly.exterior.coords)

                            if coords[0] != coords[-1]:
                                coords.append(coords[0])

                            smoothed_points = []

                            for i in range(len(coords) - 2):
                                p0 = np.array(coords[i])
                                p1 = np.array(coords[i+1])
                                p2 = np.array(coords[i+2])

                                curve = bezier_curve(p0, p1, p2, n)
                                smoothed_points.extend(curve[:-1]) #Pour éviter les doublons

                            smoothed_points.append(smoothed_points[0])

                            smooth_poly = Polygon(smoothed_points)

                            return orient(smooth_poly)

                        rim_approx_smooth_geom = smooth_polygon_with_bezier(rim_approx_geom)
                        '''

                        ### --- CREATING TOPOGRAPHIC PROFILES --- ###

                        main(demi_profils_value, demi_profils_coords_relatives, pixel_size_tb, swirl_on_or_off,
                             zone, crater_id)

                        ### --- TRI ALGORITHM --- ###
                        TRI(center_x, center_y, radius, src, no_data_value, pixel_size_tb, crater_id, zone, craters.crs)

                        ### --- SLOPES CALCULATION --- ###
                        slopes, delta_slopes = slopes_calculation(min_pos, min_val, max_value, max_coord_relative,
                                                                  pixel_size_tb, precision_error)

                        slopes_PCA, delta_slopes_PCA = slope_calculation_by_PCA(demi_profils_value,
                                                                                demi_profils_coords_relatives,
                                                                                index_maximum, out_transform,
                                                                                pixel_size_tb, precision_error,
                                                                                n_simulations=100, visualize=False)

                        # Commune attributes
                        common_attrs = {
                            'run_id': crater_id,
                            'NAC_DTM_ID': nac_id
                        }

                        # Line (profiles)
                        angle = 0
                        for i, geom in enumerate(line_geom):
                            Lines_visualisation.append({
                                'geometry': geom,
                                'position': f'Ligne à {angle}°',
                                'slope': slopes[i],
                                'δ_slope': delta_slopes[i],
                                'slopesPCA': slopes_PCA[i],
                                'δ_PCA': delta_slopes_PCA[i],
                                **common_attrs
                            })

                            highest_points.append({
                                'geometry': max_geom[i],
                                'long': max_coord_real[i][0],
                                'lat': max_coord_real[i][1],
                                'max_alt': round(max_value[i], 1),
                                'position': f'Point à {angle}°',
                                **common_attrs
                            })

                            angle += 10

                        # Crater's center
                        centers.append({
                            'geometry': coord_center_geom,
                            'center_lon': center_x,
                            'center_lat': center_y,
                            **common_attrs
                        })

                        # Lowest point
                        lowest_points.append({
                            'geometry': min_geom,
                            'alt': round(min_val, 1),
                            'position': lowest_point_coord,
                            **common_attrs
                        })

                        # Approximative rim
                        rim_approx.append({
                            'geometry': rim_approx_geom,
                            **common_attrs
                        })

                        # Crater's metadata
                        result_geom_select_crat.append({
                            'geometry': buf_diam_max,
                            'center_lon': center_x,
                            'center_lat': center_y,
                            'ray_maxdia': ray_largest_diam,
                            'moyen_diam': int(moy_diam),
                            'δ_D_stop': round(delta_Dbarre_stopar, 0),
                            'δ_D_hoov': round(delta_D_hoover, 0),
                            'prof_moyen': round(prof_moyen_crat, 1),
                            'δ_d_stop': round(delta_d_stopar, 1),
                            'δ_d_hoov': round(delta_d_hoover, 1),
                            'ratio_dD': ratio_dD,
                            'δ_dD_stop': delta_dD_stopar,
                            'δ_dD_hoov': delta_dD_hoover,
                            'circu': circularity,
                            'pente_rim': max_slope_crater,
                            'swirl': swirl_on_or_off,
                            'hiesinger': floor_age,
                            **common_attrs
                        })

                        # Point for the 90° profiles
                        profil_90.extend([
                            {
                                'geometry': point_haut_vert_360,
                                'max_alt': round(max_val_top, 1),
                                'position': "point haut",
                                **common_attrs
                            },
                            {
                                'geometry': point_haut_vert_180,
                                'max_alt': round(max_val_bas, 1),
                                'position': "point bas",
                                **common_attrs
                            },
                            {
                                'geometry': point_haut_horiz_270,
                                'max_alt': round(max_val_left, 1),
                                'position': "point gauche",
                                **common_attrs
                            },
                            {
                                'geometry': point_haut_horiz_90,
                                'max_alt': round(max_val_right, 1),
                                'position': "point droite",
                                **common_attrs
                            }
                        ])

########################################################################################################################
##################################################### RESULTS DATA #####################################################
########################################################################################################################

# List of the wanted shapefiles
shapefile_data = [
    (result_geom_select_crat,  f'results_geom_08_40m_RG{zone}_v2'),
    (lowest_points,            f'results_geom_bas_RG{zone}'),
    (highest_points,           f'results_geom_max_RG{zone}'),
    (profil_90,                f'results_geom_90_RG{zone}'),
    (Lines_visualisation,      f'results_geom_line_RG{zone}'),
    (centers,                  f'results_geom_centers_RG{zone}'),
    (rim_approx,               f'results_geom_rim_RG{zone}')
]

# Création et export des GeoDataFrames
for data, filename in shapefile_data:
    gdf = gpd.GeoDataFrame(data, crs=craters.crs)
    shapefile_path = f'results/RG{zone}/{filename}.shp'
    gdf.to_file(shapefile_path)

'''
gdf_rim_smoothed = gpd.GeoDataFrame(rim_approx_smooth, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_rim_smoothed_RG' + zone + '.shp'
gdf_rim_smoothed.to_file(shapefile_path)
'''
