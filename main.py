########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################

import geopandas as gpd
import os
import shutil
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from tqdm import tqdm
from Maximum_search import find_maxima
from Circularity import Miller_index
from Slopes import max_crater_slopes_calculation, slopes_stopar_calculation
from TRI import TRI
from Topographical_profiles import main, visualisation3d
from PDF_report import create_crater_report


########################################################################################################################
##################################################### DATA OPENING #####################################################
########################################################################################################################

zones = [7]

# Definition of the pixel size and of the vertical precision error for each zone (DTM)
zone_settings = {
    2: {'pixel_size_tb': 2, 'precision_error': 0.81},
    3: {'pixel_size_tb': 2, 'precision_error': 0.91},
    4: {'pixel_size_tb': 2, 'precision_error': 0.87},
    5: {'pixel_size_tb': 5, 'precision_error': 2.54},
    6: {'pixel_size_tb': 5, 'precision_error': 2.34},
    7: {'pixel_size_tb': 5, 'precision_error': 2.37},
    8: {'pixel_size_tb': 5, 'precision_error': 1.89}
}

for zone in zones:

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

    profile_90 = []                 # Stores the geometry of the highest points of the ridge every 90°

    centers = []                    # Stores the geometry of the centers of each valid crater found by YOLOv5

    # LINESTRING geometry

    results_slopes = []

    # POLYGON geometry

    rim_approx = []                 # Stores the geometry resulting from the polygon formed by the highest_points

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
    folders = ["profiles", "TRI", "crater_img", "reports"]
    base_path = f"results/RG{zone}"

    for folder in folders:
        path = os.path.join(base_path, folder)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"Error removing {path}: {e.strerror}")

    # Open raster file
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

            coord_center_geom = Point(coord_center)

            # Cut of the DTM with the crater
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
                continue  # No data for this crater

            floor_age = hiesinger.loc[matches_hiesinger, 'Model_Age'].values[0]

            ### --- SWIRL --- ###
            crater_center = crater_center.to_crs(swirls.crs)
            matches_swirl = swirls.geometry.contains(crater_center.geometry.iloc[0])
            swirl_on_or_off = 'on-swirl' if matches_swirl.any() else 'off-swirl'

            ### --- LOW ELEVATION --- ###
            if masked_image.count() > 0:
                min_val = round(masked_image.min(), 4)
                min_pos = np.unravel_index(masked_image.argmin(), masked_image.shape)

            ### --- HIGH ELEVATION --- ###

            (lowest_point_coord,
             min_geom,
             not_enough_data,
             max_value,
             max_coord_relative,
             max_coord_real,
             max_geom,
             demi_profiles_value,
             demi_profiles_coords_relatives
             ) = find_maxima(min_pos, min_val, masked_image, out_transform)

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

                def barycenter(points):
                    n = len(points)

                    A = 0
                    Cx = 0
                    Cy = 0

                    for i in range (n):
                        x0, y0 = points[i]
                        x1, y1 = points[(i+1) % n]
                        cross = x0 * y1 - x1 * y0
                        A += cross
                        Cx += (x0 + x1) * cross
                        Cy += (y0 + y1) * cross
                    A = A * 0.5

                    if A == 0:
                        raise ValueError("The polygon area is null")

                    Cx /= (6 * A)
                    Cy /= (6 * A)

                    return Cx, Cy

                x_bary, y_bary = barycenter(max_coord_real)

                dist_lowest_point_center = np.sqrt(
                    (x_bary - lowest_point_coord[0])**2
                    + (y_bary - lowest_point_coord[1])**2
                )

                geom_bary = Point([x_bary, y_bary])

                if dist_lowest_point_center < moy_diam * 0.25 and moy_diam >= 40:

                    ### --- CIRCULARITY --- ###
                    circularity = Miller_index(max_coord_real)
                    circularity = round(circularity, 2)

                    print("✅ Circularity calculation done")

                    if 0.98 <= circularity <= 1:

                        ### --- SLOPES --- ###

                        max_slope_crater = max_crater_slopes_calculation(max_value, max_coord_relative, pixel_size_tb)

                        if max_slope_crater < 8:

                            print(f"✅ The maximum slope between to opposite point on the rim is {max_slope_crater}")

                            ### --- CREATION OF A CIRCLE ADJUSTED TO CRATER DIMENSIONS --- ###

                            def buffer_diam_max(center_x, center_y, radius, num_points=40):
                                center = Point(center_x, center_y)
                                buffer_poly = center.buffer(radius, resolution=num_points)
                                return buffer_poly

                            buf_diam_max = buffer_diam_max(x_bary, y_bary, ray_largest_diam)

                            ### --- AVERAGE CRATER DEPTH --- ###

                            depth = [x - min_val for x in max_value]

                            prof_moyen_crat = round(np.mean(depth), 3)

                            sigma = np.sqrt(precision_error ** 2 + np.std(depth) ** 2)
                            delta_d_hoover = sigma / np.sqrt(N + 1)                         # Hoover et al., 2024

                            ### --- d//D CALCULATION --- ###

                            ratio_dD = round(prof_moyen_crat / moy_diam, 3)

                            # d/D UNCERTAINTIES CALCULATION

                            ## Hoover et al., 2024
                            rel_err_prof_hoover = delta_d_hoover / prof_moyen_crat
                            rel_err_diam_hoover = delta_D_hoover / moy_diam
                            rel_err_ratio_hoover = np.sqrt(rel_err_prof_hoover ** 2 + rel_err_diam_hoover ** 2)
                            delta_dD_hoover = round(rel_err_ratio_hoover * ratio_dD, 3)

                            print("✅ d/D done")

        ### Add geometry from highest_points
                            rim_approx_geom = Polygon(max_coord_real)
                            
                            ### --- CREATING TOPOGRAPHIC PROFILES --- ###

                            crater_floor, point_inner, idx_inner, crater_morph = main(demi_profiles_value,
                                                                        demi_profiles_coords_relatives,
                                                                        pixel_size_tb, swirl_on_or_off, zone, crater_id,
                                                                        no_data_value, depth, min_val)

                            ### --- SLOPES CALCULATION --- ###

                            visualisation3d(masked_image, crater_id, zone, swirl_on_or_off)

                            (
                                slopes_stopar,
                                slopes_stopar_geom,
                                mean_slope_stopar,
                                delta_stopar,
                                rim_height,
                                reliability_rim_height
                            ) = slopes_stopar_calculation(
                                demi_profiles_value,
                                demi_profiles_coords_relatives,
                                max_coord_real,
                                max_value,
                                point_inner,
                                idx_inner,
                                crater_floor,
                                pixel_size_tb,
                                precision_error,
                                out_transform,
                                no_data_value,
                                zone
                            )

                            print("✅ Slopes calculation done")

                            ### --- TRI ALGORITHM --- ###
                            TRI(center_x, center_y, radius, src, no_data_value, pixel_size_tb, crater_id, zone,
                                craters.crs)

                            print("✅ TRI done")

                            ### --- AUTOMATIC CLASSIFICATION --- ###

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

                            ### --- RAPORT --- ###
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
                                                 prof_moyen_crat,
                                                 round(delta_d_hoover, 1),
                                                 ratio_dD,
                                                 delta_dD_hoover,
                                                 circularity,
                                                 mean_slope_stopar,
                                                 slopes_stopar,
                                                 delta_stopar
                                                 )


                            # Commune attributes
                            common_attrs = {
                                'run_id': crater_id,
                                'NAC_DTM_ID': nac_id
                            }

                            # Line (profiles)
                            angle = 0
                            for i, geom in enumerate(max_geom):

                                highest_points.append({
                                    'geometry': max_geom[i],
                                    **common_attrs,
                                    'long': max_coord_real[i][0],
                                    'lat': max_coord_real[i][1],
                                    'max_alt': round(max_value[i], 1),
                                    'position': f'Point à {angle}°'
                                })

                                results_slopes.append({
                                    'geometry': slopes_stopar_geom[i],
                                    **common_attrs,
                                    'position': f'Ligne à {angle}°',
                                    'slopeStopar': slopes_stopar[i],
                                    'δStopar': delta_stopar[i],
                                    'meanStopar': mean_slope_stopar,
                                    "rim_height": rim_height[i],
                                    "reliability": reliability_rim_height[i],
                                    "mean_rim_h": np.mean(rim_height),
                                    "mean_reliab": np.mean(reliability_rim_height)
                                })

                                angle += 10

                            # Crater's center
                            centers.append({
                                'geometry': geom_bary,
                                **common_attrs,
                                'center_lon': x_bary,
                                'center_lat': y_bary
                            })

                            # Lowest point
                            lowest_points.append({
                                'geometry': min_geom,
                                **common_attrs,
                                'alt': round(min_val, 1),
                                'position': lowest_point_coord
                            })

                            # Approximative rim
                            rim_approx.append({
                                'geometry': rim_approx_geom,
                                **common_attrs
                            })

                            # Crater's metadata
                            result_geom_select_crat.append({
                                'geometry': buf_diam_max,
                                **common_attrs,
                                "morphology": crater_morph,
                                "deterior": state,
                                'center_lon': center_x,
                                'center_lat': center_y,
                                'ray_maxdia': ray_largest_diam,
                                'moyen_diam': int(moy_diam),
                                'δ_D_hoov': round(delta_D_hoover, 0),
                                'prof_moyen': round(prof_moyen_crat, 1),
                                'δ_d_hoov': round(delta_d_hoover, 1),
                                'ratio_dD': ratio_dD,
                                'δ_dD_hoov': delta_dD_hoover,
                                'circu': circularity,
                                'mean_slope': mean_slope_stopar,
                                "mean_rim_h": np.mean(rim_height),
                                "mean_reliab": np.mean(reliability_rim_height),
                                'swirl': swirl_on_or_off,
                                'hiesinger': floor_age
                            })

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

    # List of the wanted shapefiles
    shapefile_data = [
        (result_geom_select_crat,       f'0_results_global_RG{zone}'),
        (rim_approx,                    f'1_results_rim_RG{zone}'),
        (results_slopes,                f'2_results_slopes_RG{zone}'),
        (highest_points,                f'3_results_max_RG{zone}'),
        (lowest_points,                 f'4_results_low_RG{zone}'),
        (centers,                       f'5_results_centers_RG{zone}')
    ]
    # Création et export des GeoDataFrames
    for data, filename in shapefile_data:
        gdf = gpd.GeoDataFrame(data, crs=craters.crs)
        shapefile_path = f'results/RG{zone}/{filename}.shp'
        gdf.to_file(shapefile_path)


