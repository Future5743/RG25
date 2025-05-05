# 11/04/2025. Code en cours de mise à jour.

######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import geopandas as gpd    # Import de la bibliothèque python "Geopandas". Permet de manipuler des données géographiques

import skimage as sk

import os

import shutil

import rasterio

from rasterio.mask import mask

import numpy as np

from shapely.geometry import Point, LineString, Polygon

from tqdm import tqdm

from Maximum_search import Finding_maxima, horizontal_90, horizontal_270, vertical_360, vertical_180

from Circularity import Miller_index

from Slopes import max_crater_slopes_calculation

from TRI import TRI

from Topographical_profiles import profils_topo

######################################################################################################################################################################################
#################################################################################### DATA OPENING ####################################################################################
######################################################################################################################################################################################

zone = '7'

if zone in ['1', '2', '3', '4']:
    pixel_size_tb = 2
else :
    pixel_size_tb = 5

# Stocke le chemin d'accès du fichier shapefile "Buffer (large) des cratères" dans une variable
crater_shapefile_path = 'data/Buffer_crateres/Buffer_RG' + zone + '/'

# Stocke le chemin d'accès du fichier raster "Modèle Numérique de Terrain" dans une variable
raster_path = "../data/DTM/NAC_DTM_REINER" + zone + ".tiff"

# Stocke le chemin d'accès du fichier shapefile HIESINGER2011_MARE_AGE_UNITS_180 dans une variable
hiesinger_path = "data/HIESINGER2011_MARE_AGE_UNITS_180/HIESINGER2011_MARE_AGE_UNITS_180.SHP"

# Stocke le chemin d'acces du fichier shapefile "LUNAR_SWIRLS_180" dans une variable
swirls_path = "data/Swirl/REINER_GAMMA.shp"

# Lecture des variables à l'aide de Géopandas.
# Devient un GeoDataFrame (permet la visualisation et la manipulation du fichier)
craters = gpd.read_file(crater_shapefile_path)
hiesinger = gpd.read_file(hiesinger_path)
swirls = gpd.read_file(swirls_path)

swirls_geom = swirls['geometry']
hiesinger_geom = hiesinger['geometry']
hiesinger_age = hiesinger['Model_Age']

######################################################################################################################################################################################
#################################################################################### LIST CREATION ###################################################################################
######################################################################################################################################################################################

# POINT geometry
highest_points = []             # Stocke la géométrie des points les plus hauts correspondants à la crête du cratère

lowest_points = []              # Stocke la géométrie des points le plus bas

profil_90 = []                  # Stocke la géométrie des points les plus hauts de la crête tous les 90°

centers = []                    # Stocke la géométrie des centres de chaque cratères valide trouvé par YOLOv5

# LINESTRING geometry

Lignes_visualisation = []       # Stocke la géométrie des lignes pour visualiser chaques profils

# POLYGON geometry

rim_approx = []                 # Stocke la géométrie issue du polygone formé par les highest_points

rim_approx_smooth = []          # Stocke la géométrie lissée issue du polygone formé par les highest_points

# Liste pour stocker les informations nécessaires concernant la pente entre les bords opposés d'un cratère
results_pente = []

# Liste pour stocker les informations nécessaires concernant la circularité d'un cratère
# et ce pour l'ensemble des cratères sélectionnés
results_circularite = []

# Géométrie des cratères finaux
result_geom_select_crat = []

# Liste pour stocker les informations nécessaires concernant le calcul final du ratio dD de chaque cratères
results_ratio_dD = []

# Force l'affichage complet d'un tableau numpy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################
if os.path.exists("results/RG" + zone + "/profils"):
    try:
        shutil.rmtree("results/RG" + zone + "/profils")
    except OSError as e:
        print(f"Error:{e.strerror}")


if os.path.exists("results/RG" + zone + "/TRI"):
    try:
        shutil.rmtree("results/RG" + zone + "/TRI")
    except OSError as e:
        print(f"Error:{e.strerror}")


# Ouverture du fichier raster.
with rasterio.open(raster_path) as src:
    # Lecture de la première bande du raster et l'intègre dans un tableau numpy contenant les valeurs des pixels.
    tableau = src.read(1)

    # Récupérer les pixel "no data" du raster.
    no_data_value = src.nodata

    # Pour chaque cratère (polygone de cercle)
    for _, crater in tqdm(craters.iterrows()):
        # Extraire la géométrie du polygone, son identifiant unique et autres données attributaires
        geometry_cr =   [crater['geometry']]
        id =            crater['run_ID']
        nac_id =        crater['NAC_DTM_ID']
        center_x_dl =   crater['center_lon']
        center_y_dl =   crater['center_lat']
        ray = crater['ray_maxdia']

        # Découper le raster en utilisant le polygone
        out_image, out_transform = mask(src, geometry_cr, crop=True)

        # Ignorer les valeurs "no data" (à confirmer si j'ai besoin de cette ligne)
        masked_image = np.ma.masked_equal(out_image, no_data_value)

        coord_center = (center_x_dl, center_y_dl)

        coord_center_geom = Point(coord_center)

### HIESINGER
        gdf_centre_crater = gpd.GeoDataFrame([({'geometry': coord_center_geom})], crs=craters.crs)

        gdf_centre_crater = gdf_centre_crater.to_crs(hiesinger.crs)

        # Si un cratère n'a pas de data de Hiesinger, on ne le prend pas en compte
        if hiesinger_geom.contains(gdf_centre_crater['geometry'].iloc[0]).any() :

            for i in range(hiesinger_geom.shape[0]):
                if hiesinger_geom.tolist()[i].contains(gdf_centre_crater['geometry'].iloc[0]):
                    floor_age = hiesinger_age[i]
                    break

    ### ON-SWIRLS or OFF-SWIRLS
            gdf_centre_crater = gdf_centre_crater.to_crs(swirls.crs)

            swirl_on_or_off = 'off-swirl'

            if swirls_geom.contains(gdf_centre_crater['geometry'].iloc[0]).any() :
                swirl_on_or_off = 'on-swirl'

        ### ELEVATION BASSE : A l'interieur du cercle, trouver l'élévation la plus basse et sa position

            if masked_image.count() > 0:
                min_val = round(masked_image.min(), 4)
                min_pos = np.unravel_index(masked_image.argmin(), masked_image.shape)

            D = masked_image.shape[1] * 2

        ### ELEVATION HAUTE

            # Initialisation de listes pour stocker les données futures
            max_value = []                      # Stocke les valeurs des altitudes des highest_points
            max_coord_relative = []             # Stocke les coordonnées relatives des highest_points
            max_coord_real = []                 # Stocke les coordonnées réelles des highest_points
            max_geom = []                       # Stocke les géométries des highest_points
            line_geom = []                      # Stocke les géométries des lignes des profils étudiés
            profils = []                        # Stocke les profls topographiques
            demi_profils_coords_relatives = []  # Stocke les coordonnées relatives des points présents dans le profil

            lowest_point_coord, min_geom = Finding_maxima(min_pos, min_val, D, masked_image, out_transform, max_value,
                                                          max_coord_relative, max_coord_real, max_geom, line_geom,
                                                          profils, demi_profils_coords_relatives)

            if len(max_geom) == 36:

        ### ELEVATION HAUTE : Horizontal - 90° (vers la droite dans le plan en 2-Dimensions)

                max_val_right, point_haut_horiz_90 = horizontal_90(min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Horizontal - 270° (vers la gauche dans le plan en 2-Dimensions)

                max_val_left, point_haut_horiz_270 = horizontal_270(min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Vertical - 360° (vers le haut dans le plan en 2-Dimensions)

                max_val_top, point_haut_vert_360 = vertical_360 (min_pos, masked_image, no_data_value, out_transform)

        # ELEVATION HAUTE : Vertical - 180° (vers le bas dans le plan en 2-Dimensions)

                max_val_bas, point_haut_vert_180 = vertical_180(min_pos, masked_image, no_data_value, out_transform)

        ### DIAMETRES
                D = []

                # Calcul des differents valeurs de diametres
                def calcul_distance(pos1, pos2, pixel_size_tb=2):

                    pixel_dist_tb = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

                    distance_in_meters_tb = pixel_dist_tb * pixel_size_tb

                    return distance_in_meters_tb

                delta_pixel = 1  # incertitude sur chaque coordonnée (en pixels)
                incertitudes_individuelles = []

                for i in range(int(len(max_coord_relative) / 2)):
                    pos1 = max_coord_relative[i]
                    pos2 = max_coord_relative[i + 18]

                    d = calcul_distance(pos1, pos2, pixel_size_tb)
                    D.append(d)

                    # incertitude sur la distance (d) en mètres
                    delta_d = pixel_size_tb * np.sqrt(2) * delta_pixel
                    incertitudes_individuelles.append(delta_d)

                D = np.array(D)
                incertitudes_individuelles = np.array(incertitudes_individuelles)

                # Moyenne des diamètres
                moy_diam = round(np.mean(D), 2)

                # Incertitude sur la moyenne
                sigma_D = np.std(D, ddof=1)
                N = len(D)
                incertitude_moy_diam = round(sigma_D / np.sqrt(N), 2)

        # Calcul du rayon de la moyenne des diamètres
                ray_largest_diam = round(moy_diam / 2, 1)

                if calcul_distance(lowest_point_coord, coord_center, pixel_size_tb) < moy_diam * 0.25 and moy_diam >= 40:

                    '''
                # Profil vertical
                    # Calcul de la distance euclidienne en pixels entre max_pos_bas et max_pos_top et la convertir en mètres
                    def calculate_pixel_distance_tb(pos1tb, pos2tb, pixel_size_tb=5):
    
                        # Calculer la distance euclidienne en pixels
                        pixel_distance_tb = np.sqrt((pos2tb[0] - pos1tb[0]) ** 2 + (pos2tb[1] - pos1tb[1]) ** 2)
    
                        # Convertir la distance en mètres
                        distance_in_meters_tb = pixel_distance_tb * pixel_size_tb
                        return distance_in_meters_tb
    
                    # Conversion de la distance euclidienne en mètres
                    distance_bas_top = calculate_pixel_distance_tb(max_pos_bas, max_pos_top)
    
                # Profil horizontal
                    # Calcul de la distance euclidienne en pixels entre max_pos_right et max_pos_left et la convertir en mètres
                    def calculate_pixel_distance_lr(pos1lr, pos2lr, pixel_size_lr=5):
    
                        # Calculer la distance euclidienne en pixels
                        pixel_distance_lr = np.sqrt((pos2lr[0] - pos1lr[0]) ** 2 + (pos2lr[1] - pos1lr[1]) ** 2)
    
                        # Conversion de la distance euclidienne en mètres
                        distance_in_meters_lr = pixel_distance_lr * pixel_size_lr
                        return distance_in_meters_lr
    
                    # Conversion de la distance euclidienne en mètres
                    distance_right_left = calculate_pixel_distance_lr(max_pos_right, max_pos_left)
                    '''

                # Calcul de l'indice de circularité de Miller
                    circularity = Miller_index(min_pos, max_coord_relative, pixel_size_tb)
                    circularity = round(circularity, 2)

                    if 0.99 <= circularity <= 1:

        ### PENTE

                        max_slope_crater = max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb)

                        if max_slope_crater < 8:

                            '''
                        # Profil Horizontal
                            # Trouver la plus basse altitude et la plus haute altitude entre celle de l'Est et celle de l'Ouest
                            petite_altitude = min(max_val_right, max_val_left)
                            grande_altitude = max(max_val_right, max_val_left)
        
                            # Calculer la différence d'altitude
                            altitude_difference = round(grande_altitude - petite_altitude, 4)
        
                            # Calculer la pente en radians avec NumPy
                            slope_radians = round(np.arctan(altitude_difference / distance_right_left), 4)
        
                            # Convertir la pente en degrés
                            slope_degrees_eo = round(np.degrees(slope_radians), 4)
        
                        # Profil Vertical
                            # Trouver la plus basse altitude et la plus haute altitude entre celle du Nord et celle du Sud
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

        ### CREATION D'UN CERCLE AJUSTE AUX DIMENSIONS DU CRATERE
                            def buffer_diam_max(center_x, center_y, radius, num_points=40):
                                center = Point(center_x, center_y)
                                buffer_poly = center.buffer(radius, resolution=num_points)
                                return buffer_poly

                            buf_diam_max = buffer_diam_max(lowest_point_coord[0], lowest_point_coord[1], ray_largest_diam)

                        # # DISTANCE MIN_VAL AVEC CHACUN DES POINTS HAUTS E,O,N,S
                        #     def pixel_distance_centre_haut(pos1ch, pos2ch, pixel_size_ch=2):
                        #
                        #         # Calculer la distance euclidienne en pixels
                        #         pixel_distance_ch = np.sqrt((pos2ch[0] - pos1ch[0]) ** 2 + (pos2ch[1] - pos1ch[1]) ** 2)
                        #
                        #         # Conversion de la distance euclidienne en mètres
                        #         distance_in_meters_ch = pixel_distance_ch * pixel_size_ch
                        #         return distance_in_meters_ch
                        #
                        #     # Conversion de la distance euclidienne en mètres
                        #     distance_centre_haut = pixel_distance_centre_haut(lowest_points[0], highest_points[0])
                        #     print(f"Distance entre le point haut au Nord et le point bas au centre : {distance_centre_haut} mètres", id)

                    # PROFONDEUR MOYENNE DU CRATERE

                            moyenne_altitude = round(np.mean(max_value), 4)
                            prof_moyen_crat = round(moyenne_altitude - min_val, 3)
                            delta_prof = np.sqrt(2) * pixel_size_tb  # propagation verticale

                            ratio_dD = round(prof_moyen_crat / moy_diam, 3)
                            rel_err_prof = delta_prof / prof_moyen_crat
                            rel_err_diam = incertitude_moy_diam / moy_diam
                            rel_err_ratio = np.sqrt(rel_err_prof ** 2 + rel_err_diam ** 2)
                            delta_dD = round(rel_err_ratio * ratio_dD, 4)

                    # Ajouter les informations de Pente à la liste results_pente
                    #         results_pente.append({'run_ID': id,
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

                        # Ajouter les informations de Circularité à la liste results_circularite
                        #     results_circularite.append({'run_ID': id,
                        #                                 'Diametre_PHS_PHN': distance_bas_top,
                        #                                 'Diametre_PHE_PHO': distance_right_left,
                        #                                 'Moyenne_diametre': moy_diam,
                        #                                 'Circularite': round(circularity, 2)
                        #                                 })

                            # Teste si le pixel haut Est, Ouest, Nord et Sud ne se trouve pas trop prêt de la limite du
                            # fichier de forme de polygone "selection_crateres.shp"

        ### Ajout de la géométrie issue des highest_points
                            rim_approx_geom = Polygon(max_coord_real)

        ### Lissage du polygone issu du polygone précédent

                            '''
                            def bezier_curve(p0, p1, p2, n=20) :
                                t = np.linspace(0,1,n)

                                return (1-t)[:, None]**2 * p0 + 2 * (1-t)[:, None] * t[:, None] * p1 + t[:, None]**2 * p2

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

        ### CREATION DES PROFILS TOPOGRAPHIQUES

                            # profils_topo(profils, demi_profils_coords_relatives, pixel_size_tb, id, zone, swirl_on_or_off)

        ### ALGORITHME TRI
                            # TRI(center_x_dl, center_y_dl, ray, src, no_data_value, pixel_size_tb, id, zone, craters.crs)

        ### MISE EN PLACE DES DATAS POUR LA CREATION FUTURE DES SHAPEFILE
                            angle = 0

                            for line in line_geom:
                                Lignes_visualisation.append(({'geometry': line,
                                                              'position': str(angle),
                                                              'run_id': id,
                                                              'NAC_DTM_ID': nac_id}))
                                angle += 10

                            centers.append(({'geometry': coord_center_geom,
                                             'run_id': id,
                                             'center_lon': center_x_dl,
                                             'center_lat': center_y_dl}))

                            rim_approx.append(({'geometry': rim_approx_geom,
                                                'run_id': id,
                                                'NAC_DTM_ID': nac_id}))

                            # rim_approx_smooth.append(({'geometry': rim_approx_smooth_geom, 'run_id': id, 'NAC_DTM_ID': nac_id}))

                            result_geom_select_crat.append({'run_id': id,
                                                            'nac_dtm_id': nac_id,
                                                            'geometry': buf_diam_max,
                                                            'center_lon': center_x_dl,
                                                            'center_lat': center_y_dl,
                                                            'ray_maxdia': ray_largest_diam,
                                                            'moyen_diam': int(moy_diam),
                                                            'incer_D': incertitude_moy_diam,
                                                            'incer_d': delta_prof,
                                                            'prof_moyen': round(prof_moyen_crat, 1),
                                                            'ratio_dD': ratio_dD,
                                                            'incer_dD': delta_dD,
                                                            'circularit': circularity,
                                                            'pente_rim': max_slope_crater,
                                                            'swirl': swirl_on_or_off,
                                                            'hiesingerA': floor_age})

                            # Créer un objet Point

                            lowest_points.append({'geometry': min_geom,
                                                  'alt': round(min_val, 1),
                                                  'position': lowest_point_coord,
                                                  'run_id': id,
                                                  'NAC_DTM_ID': nac_id})

                            for i in range(len(max_geom)):

                                highest_points.append({'geometry': max_geom[i],
                                                       'long': max_coord_real[i][0],
                                                       'lat': max_coord_real[i][1],
                                                       'max_alt': round(max_value[i], 1),
                                                       'position': "point à" + str(angle) + '°',
                                                       'run_id': id,
                                                       'NAC_DTM_ID': nac_id})

                                angle += 10

                            profil_90.append({ 'geometry': point_haut_vert_360,    'max_alt': round(max_val_top, 1),
                                               'position': "point haut",   'run_id': id, 'NAC_DTM_ID': nac_id})
                            profil_90.append({ 'geometry': point_haut_vert_180,    'max_alt': round(max_val_bas, 1),
                                               'position': "point bas",    'run_id': id, 'NAC_DTM_ID': nac_id})
                            profil_90.append({ 'geometry': point_haut_horiz_270,   'max_alt': round(max_val_left, 1),
                                               'position': "point gauche", 'run_id': id, 'NAC_DTM_ID': nac_id})
                            profil_90.append({ 'geometry': point_haut_horiz_90,    'max_alt': round(max_val_right, 1),
                                               'position': "point droite", 'run_id': id, 'NAC_DTM_ID': nac_id})


######################################################################################################################################################################################
#################################################################################### RESULTS DATA ####################################################################################
######################################################################################################################################################################################

# Création des fichiers finaux

# Créer un GeoDataFrame avec les coordonnées GPS des cercles répondant à l'ensemble des critères de sélection des cratères pour l'étude du ratio d/D
gdf = gpd.GeoDataFrame(result_geom_select_crat, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_08_40m_RG' + zone + '_v2.shp'
gdf.to_file(shapefile_path)

# Créer un GeoDataFrame avec les coordonnées GPS des points bas
gdf_haut_rg = gpd.GeoDataFrame(lowest_points, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_bas_RG' + zone + '.shp'
gdf_haut_rg.to_file(shapefile_path)

# Créer un GeoDataFrame avec les coordonnées GPS des points hauts
gdf_max = gpd.GeoDataFrame(highest_points, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_max_RG' + zone + '.shp'
gdf_max.to_file(shapefile_path)

# Créer un GeoDataFrame avec les coordonnées GPS des points hauts tous les 90*
gdf_max_90 = gpd.GeoDataFrame(profil_90, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_90_RG' + zone + '.shp'
gdf_max_90.to_file(shapefile_path)

# Pente en CSV
# Convertir la liste results_pente en DataFrame pandas
# df_pente = pd.DataFrame(results_pente)
# Sauvegarder le DataFrame dans un fichier CSV
# df_pente.to_csv('C:/Users/calg2564/PycharmProjects/pythonProject/rg2-3-4_MNT5m/results_pente_rg2.csv', index=False)

# Circularité en CSV
# Convertir la liste results_circularite en DataFrame pandas
# df_circu = pd.DataFrame(results_circularite)
# Sauvegarder le DataFrame dans un fichier CSV
# df_circu.to_csv('C:/Users/calg2564/PycharmProjects/pythonProject/rg2-3-4_MNT5m/results_circularite_rg2.csv', index=False)

gdf_line = gpd.GeoDataFrame(Lignes_visualisation, crs=craters.crs)
shapefile_path = 'results/RG' + zone + '/results_geom_line_RG' + zone + '.shp'
gdf_line.to_file(shapefile_path)

gdf_centers = gpd.GeoDataFrame(centers, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_centers_RG' + zone + '.shp'
gdf_centers.to_file(shapefile_path)

gdf_rim = gpd.GeoDataFrame(rim_approx, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_rim_RG' + zone + '.shp'
gdf_rim.to_file(shapefile_path)

'''
gdf_rim_smoothed = gpd.GeoDataFrame(rim_approx_smooth, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'results/RG' + zone + '/results_geom_rim_smoothed_RG' + zone + '.shp'
gdf_rim_smoothed.to_file(shapefile_path)
'''
