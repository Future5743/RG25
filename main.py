########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################

# Libraries
import geopandas as gpd
import numpy as np
import os
import shutil
import rasterio
from datetime import datetime
from rasterio.mask import mask
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Python files
from Circularity_and_barycenter import Miller_index, barycenter
from Maximum_search import find_maxima
from PDF_report import create_crater_report
from Slopes import max_crater_slopes_calculation, slopes_stopar_calculation
from Topographical_profiles import main
from TRI import TRI
from Wanted_morph import name_user, ask_wanted_morph, data_recovery

########################################################################################################################
##################################################### DATA OPENING #####################################################
########################################################################################################################

# This list contains all the RG that need to be analysed
zones = [6]

# Definition of the pixel size and of the vertical precision error for each zone (DTM)
# The values are from README.md files that support DTM
zone_settings = {
    2: {'pixel_size_tb': 2, 'precision_error': 0.81},
    3: {'pixel_size_tb': 2, 'precision_error': 0.91},
    4: {'pixel_size_tb': 2, 'precision_error': 0.87},
    5: {'pixel_size_tb': 5, 'precision_error': 2.54},
    6: {'pixel_size_tb': 5, 'precision_error': 2.34},
    7: {'pixel_size_tb': 5, 'precision_error': 2.37},
    8: {'pixel_size_tb': 5, 'precision_error': 1.89}
}

# User name
user_initials = name_user()

# Beginning of the process by zone
for zone in zones:

    # Ask the user what type of study he wants to perform
    wanted_morph, selected_files = ask_wanted_morph()

    # Recovering the right parameters for the area under study
    try:
        params = zone_settings.get(zone)
        pixel_size_tb = params['pixel_size_tb']
        precision_error = params['precision_error']

    except OSError as e:
        # If the zone is not recognized
        print(f"Error: {e.strerror}, invalid zone")

    # Path for data
    crater_shapefile_path = os.path.join('data', 'Buffer_crateres', f'Buffer_RG{zone}')
    raster_path = os.path.join('..', 'data', 'RG', 'DTM', f'NAC_DTM_REINER{zone}.tiff')
    hiesinger_path = os.path.join('data', 'HIESINGER2011_MARE_AGE_UNITS_180', 'HIESINGER2011_MARE_AGE_UNITS_180.SHP')
    swirls_path = os.path.join('data', 'Swirl', 'REINER_GAMMA.shp')

    # Loading shapefiles in GeoDataFrame
    craters = gpd.read_file(crater_shapefile_path)
    hiesinger = gpd.read_file(hiesinger_path)
    swirls = gpd.read_file(swirls_path)

    # If the user have chosen to reuse older data, the following extra dataframe are needed
    global_results_shp = None

    if selected_files is not None:
        for path in selected_files:
            if "global" in path:
                global_results_shp = gpd.read_file(path)
            elif "centers" in path:
                centers_shp = gpd.read_file(path)
            elif "highest" in path:
                highest_shp = gpd.read_file(path)
            elif "lowest" in path:
                lowest_shp = gpd.read_file(path)
            elif "rim" in path:
                rim_shp = gpd.read_file(path)
            elif "slopes" in path:
                slopes_shp = gpd.read_file(path)
            else:
                raise ValueError("You didn't choose the right files : you need all the .shp of your study which only"
                                 " focused on bowl-shaped craters (global-results, centers, lowest_points, "
                                 "highest_points, rim and slopes")

    # Geometry and ages extraction
    swirls_geom = swirls.geometry
    hiesinger_geom = hiesinger.geometry
    hiesinger_age = hiesinger['Model_Age']

########################################################################################################################
##################################################### LIST CREATION ####################################################
########################################################################################################################

    # POINT geometry
    highest_points = []             # Stores the geometry of the highest points corresponding to the crater crest

    lowest_points = []              # Stores the geometry of the lowest points

    centers = []                    # Stores the geometry of the centers of each valid crater found by YOLOv5

    # LINESTRING geometry

    results_slopes = []

    # POLYGON geometry

    rim_approx = []                 # Stores the geometry resulting from the polygon formed by the highest_points

    result_geom_select_crat = []    # Stores the geometry of the buffer that is created around the crater center with
                                    # the mean diameter

    # Force full display of a numpy array
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################

    ### --- CLEANING OF THE WORK ENVIRONMENT --- ###
    # This cleaning is effective only if the user doesn't want to use older data
    if global_results_shp is None:
        folders = ["profiles", "TRI", "crater_img", "reports"]
        base_path = f"results/RG{zone}"

        for folder in folders:
            path = os.path.join(base_path, folder)
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    print(f"Error removing {path}: {e.strerror}")


    ### --- OPEN RASTER FILE --- ###
    with rasterio.open(raster_path) as src:
        raster_array = src.read(1)
        no_data_value = src.nodata

        for _, crater in tqdm(craters.iterrows(), total=len(craters)):
            # Crater's data
            crater_geom = [crater.geometry]
            crater_id = crater.run_ID
            nac_id = crater.NAC_DTM_ID
            center_x, center_y = crater.center_lon, crater.center_lat
            radius = crater.ray_maxdia
            coord_center = (center_x, center_y)

            # If the crater studied has already been the subject of an earlier study, the data is reused.
            if global_results_shp is not None:

                if crater_id in global_results_shp['run_id'].values:

                    data_recovery(global_results_shp, rim_shp, centers_shp, lowest_shp, highest_shp, slopes_shp,
                                  crater_id, nac_id,
                                  highest_points, results_slopes, result_geom_select_crat, centers, lowest_points,
                                  rim_approx)

                    print(f"💾 Data for the crater {crater_id} was recovered from older versions")

                    # All the treatments are already done, we can switch to analyse another crater
                    continue

            ### --- CUT OF THE DTM WITH THE CRATER GEOMETRY --- ###
            try:
                out_image, out_transform = mask(src, crater_geom, crop=True)
            except ValueError:
                continue  # Skip invalid geometries or ones falling outside raster extent

            # Hide no_data values
            masked_image = np.ma.masked_equal(out_image, no_data_value)

            # Geodataframe for the crater center
            crater_center = gpd.GeoDataFrame(
                [{'geometry': Point(center_x, center_y)}],
                crs=craters.crs
            )

            ### --- HIESINGER --- ###
            crater_center = crater_center.to_crs(hiesinger.crs)

            # Check if the crater is in a Heisinger polygon
            matches_hiesinger = hiesinger.geometry.contains(crater_center.geometry.iloc[0])

            if not matches_hiesinger.any():
                continue  # If the crater is not contained in a Hiesinger polygon, it is skipped and thus not studied

            floor_age = hiesinger.loc[matches_hiesinger, 'Model_Age'].values[0]

            ### --- SWIRL --- ###
            crater_center = crater_center.to_crs(swirls.crs)
            matches_swirl = swirls.geometry.contains(crater_center.geometry.iloc[0])
            swirl_on_or_off = 'on-swirl' if matches_swirl.any() else 'off-swirl'    # Assigning the on or off swirl data

            ### --- LOW ELEVATION --- ###
            if masked_image.count() > 0:
                min_val = round(masked_image.min(), 4)                                  # Lowest elevation encountered
                min_pos = np.unravel_index(masked_image.argmin(), masked_image.shape)   # Relative coordinates of the
                                                                                        # lowest point

            ### --- HIGH ELEVATION --- ###

            (lowest_point_coord,                                                        # Finding of the points with the
             min_geom,                                                                  # highest elevation every 10°
             not_enough_data,                                                           # around the point with the
             max_value,                                                                 # lowest elevation
             max_coord_relative,                                                        # Also stores teh semi-profiles
             max_coord_real,                                                            # for future treatments
             max_geom,
             demi_profiles_value,
             demi_profiles_coords_relatives
             ) = find_maxima(min_pos, min_val, masked_image, out_transform)

            # First condition: the crater must have 36 semi-profile (so not near the edge of the DTM) and each
            # semi-profiles can't be cut with no data
            if len(max_geom) == 36 and not_enough_data == 0:

                print("✅ The maximum points calculation done")

                ### --- DIAMETERS --- ###
                D = []

                for i in range(int(len(max_coord_relative) / 2)):
                    pos1 = max_coord_relative[i]
                    pos2 = max_coord_relative[i + 18]

                    d = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * pixel_size_tb
                    D.append(d)

                D = np.array(D)

                # Average diameters
                moy_diam = round(np.mean(D), 2)

                # Uncertainty of a diameter

                delta_D_hoover = np.sqrt(np.std(D)**2 + pixel_size_tb**2)                   # Hoover et al., 2024

                # Uncertainty of average
                N = len(D)

                delta_Dbarre_hoover = round(delta_D_hoover / np.sqrt(N), 2)                 # Hoover et al., 2024

                # Calculation of the radius of the average diameter
                ray_largest_diam = round(moy_diam / 2, 1)

                print("✅ The diameter calculation done")

                ### --- BARYCENTER --- ###

                x_bary, y_bary = barycenter(max_coord_real)

                dist_lowest_point_center = np.sqrt(
                    (x_bary - lowest_point_coord[0])**2
                    + (y_bary - lowest_point_coord[1])**2
                )

                geom_bary = Point([x_bary, y_bary])

                # Second condition: the crater must have a diameter of at least 40m and the distance between his
                # barycenter and his minimum elevation has to be less than a quarter of the mean diameter
                if dist_lowest_point_center < moy_diam * 0.25 and moy_diam >= 40:

                    ### --- CIRCULARITY --- ###
                    circularity = Miller_index(max_coord_real)
                    circularity = round(circularity, 2)

                    print("✅ Circularity calculation done")

                    # Third condition: the crater must have a circularity of at least 0.9.
                    # For reminder, if a geometry have a Miller index of 1, it is circular, and if 0 it is far from
                    # circular
                    if 0.90 <= circularity <= 1:

                        ### --- MAX SLOPE BETWEEN TWO OPPOSITE POINTS ON THE RIM --- ###

                        max_slope_crater = max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb)

                        # Fourth condition: the maximum slope between two opposite points on the rim must be less than
                        # 8° (Stopar et al., 2017)
                        if max_slope_crater < 8:

                            print(f"✅ The maximum slope between to opposite point on the rim is {max_slope_crater}")

                            ### --- AVERAGE CRATER DEPTH --- ###

                            depth = [x - min_val for x in max_value]

                            mean_crat_depth = round(np.mean(depth), 3)

                            sigma = np.sqrt(precision_error ** 2 + np.std(depth) ** 2)
                            delta_d_hoover = sigma / np.sqrt(N)  # Hoover et al., 2024
                            
                            ### --- CREATING TOPOGRAPHIC PROFILES --- ###

                            crater_floor, point_inner, idx_inner, crater_morph = main(demi_profiles_value,
                                                                                      demi_profiles_coords_relatives,
                                                                                      pixel_size_tb,
                                                                                      swirl_on_or_off,
                                                                                      zone,
                                                                                      crater_id,
                                                                                      no_data_value,
                                                                                      depth,
                                                                                      min_val,
                                                                                      wanted_morph)

                            # Fith condition: if a crater is labeled with an "Other" morphology, it is not studied
                            # anymore. A crater can be attributed this morphology only when the algorithm is run for
                            # "Bowl-shaped only" morphology
                            if crater_morph != "Other":

                                ### --- CREATION OF A CIRCLE ADJUSTED TO CRATER DIMENSIONS --- ###

                                def buffer_diam_max(center_x, center_y, radius, num_points=40):
                                    center = Point(center_x, center_y)
                                    buffer_poly = center.buffer(radius, resolution=num_points)
                                    return buffer_poly


                                buf_diam_max = buffer_diam_max(x_bary, y_bary, ray_largest_diam)

                                ### --- d/D CALCULATION --- ###

                                ratio_dD = round(mean_crat_depth / moy_diam, 3)

                                # d/D UNCERTAINTIES CALCULATION
                                # The mathematical formula is obtained by the uncertainty propagation formula
                                rel_err_prof_hoover = delta_d_hoover / mean_crat_depth
                                rel_err_diam_hoover = delta_D_hoover / moy_diam
                                rel_err_ratio_hoover = np.sqrt(rel_err_prof_hoover ** 2 + rel_err_diam_hoover ** 2)
                                delta_dD_hoover = round(rel_err_ratio_hoover * ratio_dD, 3)

                                print("✅ d/D done")

                                ### --- TRI ALGORITHM --- ###
                                # The formula is taken from Argwal et al., 2019
                                TRI_mean_crest = TRI(center_x, center_y, radius, src, no_data_value, pixel_size_tb,
                                                     crater_id, zone, craters.crs, max_coord_real)

                                print("✅ TRI done")

                                ### --- SLOPES CALCULATION --- ###
                                # The method is taken from Stopar et al., 2017
                                (
                                    slopes_stopar,
                                    slopes_stopar_geom,
                                    mean_slope_stopar,
                                    delta_stopar
                                ) = slopes_stopar_calculation(
                                    demi_profiles_value,
                                    demi_profiles_coords_relatives,
                                    point_inner,
                                    idx_inner,
                                    crater_floor,
                                    pixel_size_tb,
                                    precision_error,
                                    out_transform,
                                    no_data_value
                                )

                                print("✅ Slopes calculation done")

                                ### --- AUTOMATIC CLASSIFICATION OF THE ESTIMATED STATE OF DETERIORATION --- ###
                                # The criteria for the estimation of the degradation state are taken from
                                # Basilevsky et al., 1976

                                state = "Unknown"
                                slope_max = np.max(slopes_stopar)

                                if ratio_dD > 1 / 5 and slope_max > 35:
                                    state = "A"
                                elif 1 / 7 < ratio_dD <= 1 / 5 and 25 < slope_max <= 35:
                                    state = "AB"
                                elif 1 / 10 < ratio_dD <= 1 / 7 and 15 < slope_max <= 25:
                                    state = "B"
                                elif 1/12 < ratio_dD < 1 / 10 and 10 < slope_max <= 15:
                                    state = "BC"
                                elif ratio_dD < 1 / 12 and slope_max < 10:
                                    state = "C"
                                elif ratio_dD > 1/7 and slope_max > 25:
                                    state = "A - AB"
                                elif 1/10 < ratio_dD <= 1/5 and 15 < slope_max <= 35:
                                    state = "AB - B"
                                elif 1/12 < ratio_dD < 1/7 and 10 < slope_max < 25:
                                    state = "B - BC"
                                elif ratio_dD < 1 / 10 and slope_max < 15:
                                    state = "BC - C"

                                print(f"〽️Degradation : {state}")

                                ### --- PDF REPORT CREATION --- ###
                                create_crater_report(crater_id,
                                                     zone,
                                                     swirl_on_or_off,
                                                     crater_morph,
                                                     state,
                                                     x_bary,
                                                     y_bary,
                                                     lowest_point_coord,
                                                     moy_diam,
                                                     round(delta_D_hoover, 0),
                                                     mean_crat_depth,
                                                     round(delta_d_hoover, 1),
                                                     ratio_dD,
                                                     delta_dD_hoover,
                                                     circularity,
                                                     slopes_stopar,
                                                     delta_stopar,
                                                     TRI_mean_crest
                                                     )

                                ### --- DATA INPUT FOR SHAPEFILEs CREATION --- ###
                                # Commune attributes
                                common_attrs = {
                                    'run_id': crater_id,
                                    'NAC_DTM_ID': nac_id
                                }

                                # For angle related data
                                angle = 0
                                for i, geom in enumerate(max_geom):

                                    # Inputs for the highest points focused Shapefile
                                    highest_points.append({
                                        'geometry': max_geom[i],
                                        **common_attrs,
                                        'long': max_coord_real[i][0],
                                        'lat': max_coord_real[i][1],
                                        'max_alt': round(max_value[i], 1),
                                        'position': f'Point à {angle}°'
                                    })

                                    # Inputs for the slope focused Shapefile
                                    results_slopes.append({
                                        'geometry': slopes_stopar_geom[i],
                                        **common_attrs,
                                        'position': f'Ligne à {angle}°',
                                        'slopeStopa': slopes_stopar[i],
                                        'δStopar': delta_stopar[i],
                                        'meanStopar': mean_slope_stopar
                                    })

                                    angle += 10

                                # Inputs for the centers focused Shapefile
                                centers.append({
                                    'geometry': geom_bary,
                                    **common_attrs,
                                    'center_lon': x_bary,
                                    'center_lat': y_bary
                                })

                                # Inputs for the lowest points focused Shapefile
                                lowest_points.append({
                                    'geometry': min_geom,
                                    **common_attrs,
                                    'alt': round(min_val, 1),
                                    'position': lowest_point_coord
                                })

                                # Inputs for the rim focused Shapefile
                                rim_approx.append({
                                    'geometry': Polygon(max_coord_real),
                                    **common_attrs
                                })

                                # Inputs for the general-results Shapefile
                                result_geom_select_crat.append({
                                    'geometry': buf_diam_max,
                                    **common_attrs,
                                    "morphology": crater_morph,
                                    "deterior": state,
                                    'center_lon': center_x,
                                    'center_lat': center_y,
                                    'ray_maxdia': ray_largest_diam,
                                    'mean_diam': int(moy_diam),
                                    'δ_D': round(delta_D_hoover, 0),
                                    'mean_depth': round(mean_crat_depth, 1),
                                    'δ_d_1': round(delta_d_hoover, 1),
                                    'ratio_dD': ratio_dD,
                                    'δ_dD': delta_dD_hoover,
                                    'circu': circularity,
                                    'mean_slope': mean_slope_stopar,
                                    'mean TRI': round(TRI_mean_crest, 2),
                                    'swirl': swirl_on_or_off,
                                    'hiesinger': floor_age
                                })

                            else:
                                print("❌ Does not have a wanted morphology")

                        else:
                            print("❌ Does not meet the 8° condition")
                            continue
                    else:
                        print("❌ Does not respect circularity")
                        continue
                else:
                    print("❌ Does not comply with minimum diameter or distance between barycentre and min point")
                    continue

            else:
                print("❌ There are no 36 profiles")
                continue

########################################################################################################################
##################################################### RESULTS DATA #####################################################
########################################################################################################################

    # date of the day
    date = datetime.today().strftime("%Y%m%d")

    # List of the wanted shapefiles
    shapefile_data = [
        (result_geom_select_crat,       f'{date}_RGdD_{user_initials}_global_results_v1'),
        (rim_approx,                    f'{date}_RGdD_{user_initials}_rim_v1'),
        (results_slopes,                f'{date}_RGdD_{user_initials}_slopes_v1'),
        (highest_points,                f'{date}_RGdD_{user_initials}_highest_points_v1'),
        (lowest_points,                 f'{date}_RGdD_{user_initials}_lowest_points_v1'),
        (centers,                       f'{date}_RGdD_{user_initials}_centers_v1')
    ]

    # Creation et export of GeoDataFrames
    for data, filename in shapefile_data:
        gdf = gpd.GeoDataFrame(data, crs=craters.crs)
        shapefile_path = f'results/RG{zone}/{filename}.shp'
        gdf.to_file(shapefile_path)

