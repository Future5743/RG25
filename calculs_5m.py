# 11/04/2025. Code en cours de mise à jour.

# Import de la bibliothèque python "Geopandas". Permet de manipuler des données géographiques.
import geopandas as gpd

import rasterio

from rasterio.mask import mask

import numpy as np

from shapely.geometry import Point, LineString, Polygon

from tqdm import tqdm

# Stocke le chemin d'accès du fichier shapefile "Buffer (large) des cratères" dans une variable.
crater_shapefile_path = 'C:/Users/calg2564/PycharmProjects/pythonProject/'
# Lecture de la variable précédente à l'aide de Géopandas. Devient un GeoDataFrame (permet la visualisation et la manipulation du fichier)
craters = gpd.read_file(crater_shapefile_path)

# Stocke le chemin d'accès du fichier raster "Modèle Numérique de Terrain" dans une variable.
raster_path = 'C:/Users/calg2564/PycharmProjects/pythonProject/MNT_entree/NAC_DTM_REINER.TIF'

# Initialisation d'une liste pour stocker la geometry 'point' avec son information d'altitude. Direction depuis l'origine : verticale 360°
# highest_points = []

# Liste pour stocker les informations nécessaires concernant la pente entre les bords opposés d'un cratère
#results_pente = []

# Liste pour stocker les informations nécessaires concernant la circularité d'un cratère et ce pour l'ensemble des cratères sélectionnés
# results_circularite = []

# Géométrie des cratères finaux
# result_geom_select_crat = []

# Liste pour stocker les informations nécessaires concernant le calcul final du ratio dD de chaque cratères
# results_ratio_dD = []

# Liste pour stocker l'information 'on-swirl' ou 'off-swirl'
# swirl = []

# Force l'affichage complet d'un tableau numpy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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

        # Découper le raster en utilisant le polygone
        out_image, out_transform = mask(src, geometry_cr, crop=True)

        # Ignorer les valeurs "no data" (à confirmer si j'ai besoin de cette ligne)
        masked_image = np.ma.masked_equal(out_image, no_data_value)

# ELEVATION BASSE : A l'interieur du cercle, trouver l'élévation la plus basse et sa position

        if masked_image.count() > 0:
            min_val = round(masked_image.min(), 4)
            min_pos = np.unravel_index(masked_image.argmin(), masked_image.shape)

# ELEVATION HAUTE : Horizontal - 90° (vers la droite dans le plan en 2-Dimensions)

    # Pour la ligne de pixels extraite, trouver les valeurs des colonnes à droite de la position minimale dans la même ligne
        # Index de la ligne où se trouve l'altitude la plus basse
        row_index = min_pos[1]
        # Extraction de la ligne à partir de la position min_pos vers la droite
        line = masked_image[0, row_index, min_pos[2]:]

        # Filtrer les valeurs "no data" de la ligne
        line_values_rg = line[line != no_data_value]

        if line_values_rg.size > 0:
            # Trouver l'altitude la plus haute dans la ligne filtrée
            max_val_right = line_values_rg.max()  # Trouver l'altitude la plus haute dans cette ligne
            max_pos_right = (row_index, min_pos[2] + np.where(line == max_val_right)[0][0])  # Calculer la position de l'altitude la plus haute

            # Vérifier si c'est la dernière valeur de la ligne
            derniere_val_line = np.where(line == max_val_right)[0][0]
            range_dern_val_line = derniere_val_line in {len(line) - 1, len(line) - 2, len(line) - 3}

            # Convertir les indices en coordonnées réelles
            max_x_rg, max_y_rg = rasterio.transform.xy(out_transform, max_pos_right[0], max_pos_right[1])

            # Créer un objet Point
            point_haut_horiz_90 = Point(max_x_rg, max_y_rg)

            # Ajouter le point avec l'altitude la plus basse à la liste
            # highest_points.append({'geometry': point_haut_horiz_90, 'max_altitude_right': max_val_right, 'run_id': id, 'NAC_DTM_ID': nac_id})

# ELEVATION HAUTE : Horizontal - 270° (vers la gauche dans le plan en 2-Dimensions)
        # Trouver les valeurs des colonnes à droite de la position minimale dans la même ligne
        line_lf = masked_image[0, row_index, :min_pos[2] + 1]  # Extraire la ligne à partir de la position min_pos vers la gauche

        # Filtrer les valeurs "no data" de la ligne
        line_values_lf = line_lf[line_lf != no_data_value]

        if line_values_lf.size > 0:
            # Trouver l'altitude la plus haute dans la ligne filtrée
            max_val_left = line_values_lf.max()  # Trouver l'altitude la plus haute dans cette ligne
            max_pos_left = (row_index, np.where(line_lf == max_val_left)[0][0])  # Calculer la position de l'altitude la plus haute

            # Vérifier si c'est la première valeur de la ligne
            premiere_val_line = np.where(line_lf == max_val_left)[0][0]
            range_prem_val_line = premiere_val_line in {0, 1, 2}

            # Convertir les indices en coordonnées réelles
            max_x_lf, max_y_lf = rasterio.transform.xy(out_transform, max_pos_left[0], max_pos_left[1])

            # Créer un objet Point
            point_haut_horiz_270 = Point(max_x_lf, max_y_lf)

# ELEVATION HAUTE : Vertical - 360° (vers le haut dans le plan en 2-Dimensions)
        # Trouver les valeurs des colonnes en haut de la position minimale dans la même colonne
        col_top_index = min_pos[2]  # Index de la colonne où se trouve l'altitude la plus basse
        col_top = masked_image[0, :min_pos[1] + 1, col_top_index]  # Extraire la colonne à partir de la position min_pos vers le haut

        # Filtrer les valeurs "no data" de la ligne
        col_values_top = col_top[col_top != no_data_value]

        if col_values_top.size > 0:
            # Trouver l'altitude la plus haute dans la ligne filtrée
            max_val_top = col_values_top.max()  # Trouver l'altitude la plus haute dans cette ligne
            max_pos_top = (np.where(col_top == max_val_top)[0][0], col_top_index)  # Calculer la position de l'altitude la plus haute

            # Vérifier si c'est la première valeur de la colonne
            premiere_val_col = np.where(col_top == max_val_top)[0][0]
            range_prem_val_col = premiere_val_col in {0, 1, 2}

            # Convertir les indices en coordonnées réelles
            max_x_top, max_y_top = rasterio.transform.xy(out_transform, max_pos_top[0], max_pos_top[1])

            # Créer un objet Point
            point_haut_vert_360 = Point(max_x_top, max_y_top)

            # Ajouter le point avec l'altitude la plus basse à la liste
            # highest_points.append({'geometry': point_haut_vert_360, 'max_altitude_right': max_val_top, 'run_id': id, 'NAC_DTM_ID': nac_id})

# ELEVATION HAUTE : Vertical - 180° (vers le bas dans le plan en 2-Dimensions)
        # Trouver les valeurs des lignes en dessous de la position minimale dans la même colonne
        col_bas_index = min_pos[2]  # Index de la colonne où se trouve l'altitude la plus basse
        col_bas = masked_image[0, min_pos[1]:, col_bas_index]  # Extraire la colonne à partir de la position min_pos vers le bas

        # Filtrer les valeurs "no data" de la colonne
        col_values_bas = col_bas[col_bas != no_data_value]

        if col_values_bas.size > 0:
            # Trouver l'altitude la plus haute dans la colonne filtrée
            max_val_bas = col_values_bas.max()  # Trouver l'altitude la plus haute dans cette colonne
            max_pos_bas = (min_pos[1] + np.where(col_bas == max_val_bas)[0][0], col_bas_index)  # Calculer la position de l'altitude la plus haute

            # Vérifier si c'est la dernière valeur de la colonne
            derniere_val_col = np.where(col_bas == max_val_bas)[0][0]
            range_derniere_val_col = derniere_val_col in {len(col_bas) - 1, len(col_bas) - 2, len(col_bas) - 3}

            # Convertir les indices en coordonnées réelles
            max_x_bas, max_y_bas = rasterio.transform.xy(out_transform, max_pos_bas[0], max_pos_bas[1])

            # Créer un objet Point
            point_haut_vert_180 = Point(max_x_bas, max_y_bas)

# CIRCULARITE Distance entre points haut opposés
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

    # Moyenne diamètre
        # Trouver le plus petit et le plus grand diamètre
        smallest_diameter = min(distance_right_left, distance_bas_top)
        largest_diameter = max(distance_right_left, distance_bas_top)

    # Moyennes des deux diamètres
        moy_diam = np.mean([distance_right_left, distance_bas_top])

    # Calculer la circularité
        circularity = round(smallest_diameter / largest_diameter, 2)

    # Calcul du rayon de la moyenne des diamètres
        ray_largest_diam = round(moy_diam / 2, 1)

# PENTE
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

    # Créer un cercle qui est ajusté au dimension du cratère
        def buffer_diam_max(center_x, center_y, radius, num_points=40):
            center = Point(center_x, center_y)
            buffer_poly = center.buffer(radius, resolution=num_points)
            return buffer_poly

        buf_diam_max = buffer_diam_max(center_x_dl, center_y_dl, ray_largest_diam)

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

    # Moyenne des 4 altitudes du cratère
# PROFONDEUR MOYENNE DU CRATERE
        moyenne_altitude = round(np.mean([max_val_right, max_val_left, max_val_top, max_val_bas]), 4)

        prof_moyen_crat = round(moyenne_altitude - min_val, 3)

# RATIO d/D - calcul du ratio d/D
        ratio_dD = round(prof_moyen_crat / moy_diam, 3)

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

        #Teste si le pixel haut Est, Ouest, Nord et Sud ne se trouve pas trop prêt de la limite du fichier de forme de polygone "selection_crateres.shp"
        if (range_derniere_val_col == False and range_prem_val_col == False and range_prem_val_line == False and range_dern_val_line == False):
            if (max_val_bas != min_val and max_val_top != min_val and max_val_right != min_val and max_val_left != min_val):
            # Rempli le tableau results_ratio_dD si tous les critères de sélection sont respectés
                if (moy_diam >= 40 and circularity >= 0.9 and circularity <= 1 and pente_max <= 8):
                    result_geom_select_crat.append({'run_id': id,
                                                    'nac_dtm_id': nac_id,
                                                    'geometry': buf_diam_max,
                                                    'center_lon': center_x_dl,
                                                    'center_lat': center_y_dl,
                                                    'ray_maxdia': ray_largest_diam,
                                                    'moyen_diam': round(moy_diam, 0),
                                                    'prof_moyen': prof_moyen_crat,
                                                    'ratio_dD': ratio_dD,
                                                    'circularit': circularity,
                                                    'pente_rim': pente_max,
                                                    'swirl': ""})
                    highest_points.append({ 'geometry': point_haut_vert_360,    'max_alt': round(max_val_top, 1),     'position': "point haut",   'run_id': id, 'NAC_DTM_ID': nac_id})
                    highest_points.append({ 'geometry': point_haut_vert_180,    'max_alt': round(max_val_bas, 1),     'position': "point bas",    'run_id': id, 'NAC_DTM_ID': nac_id})
                    highest_points.append({ 'geometry': point_haut_horiz_270,   'max_alt': round(max_val_left, 1),    'position': "point gauche", 'run_id': id, 'NAC_DTM_ID': nac_id})
                    highest_points.append({ 'geometry': point_haut_horiz_90,    'max_alt': round(max_val_right, 1),   'position': "point droite", 'run_id': id, 'NAC_DTM_ID': nac_id})
                    # lowest_point_origine.append({   'geometry': point_bas_origine,      'min_altitude': round(min_val, 1),                                'run_id': id, 'NAC_DTM_ID': nac_id})

# Création des fichiers finaux

# Créer un GeoDataFrame avec les coordonnées GPS des cercles répondant à l'ensemble des critères de sélection des cratères pour l'étude du ratio d/D
gdf = gpd.GeoDataFrame(result_geom_select_crat, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'C:/Users/calg2564/PycharmProjects/pythonProject/RG_5-6-7-8/deepL_RG_5-6-7-8/donnes_sorties/RG8/results_geom_08_40m_RG8_v2.shp'
gdf.to_file(shapefile_path)

# Créer un GeoDataFrame avec les coordonnées GPS des points bas
# gdf = gpd.GeoDataFrame(lowest_point_origine, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
# shapefile_path = 'C:/Users/calg2564/PycharmProjects/pythonProject/RG_5-6-7-8/deepL_RG_5-6-7-8/donnes_sorties/RG8/results_bas_RG_8.shp'
# gdf.to_file(shapefile_path)

# Créer un GeoDataFrame avec les coordonnées GPS des points hauts
gdf_haut_rg = gpd.GeoDataFrame(highest_points, crs=craters.crs)
# Enregistrer le GeoDataFrame au format Shapefile
shapefile_path = 'C:/Users/calg2564/PycharmProjects/pythonProject/RG_5-6-7-8/deepL_RG_5-6-7-8/donnes_sorties/RG8/results_haut_RG8_v2.shp'
gdf_haut_rg.to_file(shapefile_path)

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
