########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin, rowcol
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter
import geopandas as gpd
from PIL import Image
import time

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
def draw_TRI(TRI, id, zone):
    '''
    This function plots and saves the TRI index as a color image.

    Parameters:
    -----------
    TRI: array
        Contains the TRI values for the studied crater

    id: int
        ID of the studied crater

    zone: str
        Indicates the study zone of the crater (can be 1 to 7)

    Returns:
    --------
    None
    '''
    os.makedirs(f'results/RG{zone}/TRI', exist_ok=True)

    plt.figure()
    cax = plt.imshow(TRI, cmap='viridis_r', vmin=-0.25, vmax=0.25)
    plt.colorbar(cax)
    plt.title(f'TRI Index for Crater {id} in Zone RG{zone}')
    plt.savefig(f'results/RG{zone}/TRI/TRI_{id}.png')
    plt.close()


def array_to_GeoTIF(TRI, coord_left_down, pixel_size_tb, id, zone, crs):
    '''
    This function exports the TRI array as a GeoTIFF file.

    Parameters:
    -----------
    TRI: array
        TRI values for the crater

    coord_left_down: list
        Coordinates of the bottom-left pixel of the TRI matrix

    pixel_size_tb: int
        Pixel size in meters

    id: int
        ID of the crater

    zone: str
        Study zone

    crs: str
        Coordinate Reference System (CRS) of the study area

    Returns:
    --------
    None
    '''
    transform = from_origin(coord_left_down[0], coord_left_down[1] + pixel_size_tb * TRI.shape[0],
                            pixel_size_tb, pixel_size_tb)

    with rasterio.open(
        f'results/RG{zone}/TRI/TRI_{id}.tif',
        'w',
        driver='GTiff',
        height=TRI.shape[0],
        width=TRI.shape[1],
        count=1,
        dtype=TRI.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(TRI, 1)


def compute_TRI_array(image, no_data_value):
    '''
    Computes the TRI array using the mean and standard deviation of each 3x3 neighborhood.

    Parameters:
    -----------
    image: array
        Raster image array (1-band)

    no_data_value: float or int
        Value used to represent no-data in the raster

    Returns:
    --------
    array
        Array of TRI values
    '''
    image = np.where(image == no_data_value, np.nan, image)

    def tri_func(window):
        center = window[4]
        neighbors = np.delete(window, 4)
        mean_val = np.nanmean(neighbors)
        std_val = np.nanstd(neighbors)
        if np.isnan(center) or np.isnan(mean_val) or std_val == 0:
            return 0
        return (center - mean_val) / std_val

    return generic_filter(image[0], tri_func, size=3, mode='constant', cval=np.nan)


def extract_values_from_geotiff(geotiff_path, coords):
    """
    chemin_geotiff : chemin vers le fichier GeoTIFF
    coordonnees : liste de tuples (x, y) dans le même système de référence que le GeoTIFF
    """
    TRI_max = []

    with rasterio.open(geotiff_path) as src:

        for coord in coords:

            row, col = rowcol(src.transform, coord[0], coord[1])

            try:
                value = src.read(1)[row, col]
            except IndexError:
                value = None

            TRI_max.append(value)

    TRI_max_cleaned = [x if x is not None else np.nan for x in TRI_max]

    return round(np.nanmean(TRI_max_cleaned), 2)


def TRI(center_x_dl, center_y_dl, ray, src, no_data_value, pixel_size_tb, id, zone, crs, max_coord_real):
    '''
    Main function to compute the TRI index over a crater area.

    Parameters:
    -----------
    center_x_dl: float
        X coordinate of the crater center

    center_y_dl: float
        Y coordinate of the crater center

    ray: float
        Radius of the area to extract (square side = 2*ray)

    src: rasterio dataset
        The input raster source

    no_data_value: int or float
        Value representing no-data in the source

    pixel_size_tb: int
        Pixel size in meters

    id: int
        Crater ID

    zone: str
        Crater study zone (1 to 7)

    crs: str
        Coordinate reference system of the crater

    max_coord_real: list
        Real coordinates of the points on the rim crest

    Returns:
    --------
    None
    '''

    # Define a square polygon centered on the crater
    half_size = ray
    coords = [
        [center_x_dl - half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl + half_size],
        [center_x_dl - half_size, center_y_dl + half_size]
    ]
    polygon = [Polygon(coords)]

    # Clip the raster to the polygon
    out_image, out_transform = mask(src, polygon, crop=True)
    out_meta = src.meta
    nodata_value = out_meta.get("nodata", None)

    # Create mask for invalid pixels (no-data or NaN)
    mask_pixels = np.zeros_like(out_image[0], dtype=bool)

    if nodata_value is not None:
        mask_pixels |= np.any(out_image == nodata_value, axis=0)

    if np.issubdtype(out_image.dtype, np.floating):
        mask_pixels |= np.any(np.isnan(out_image), axis=0)

    # Create output directory
    output_dir = f'results/RG{zone}/crater_img'
    os.makedirs(output_dir, exist_ok=True)

    raster_image_path = f"../data/RG/Orthophoto/Orthophoto_RG{zone}.tiff"
    with rasterio.open(raster_image_path) as src_img:
        out_img, out_transform_img = mask(src_img, polygon, crop=True)
        img_meta = src_img.meta
        nodata_val_img = img_meta.get("nodata", None)

    # Initialize mask for image
    mask_pixels_img = np.zeros_like(out_img[0], dtype=bool)

    if nodata_val_img is not None:
        mask_pixels_img |= np.any(out_img == nodata_val_img, axis=0)

    if np.issubdtype(out_img.dtype, np.floating):
        mask_pixels_img |= np.any(np.isnan(out_img), axis=0)

    # Prepare the image for saving (either RGB or grayscale)
    if out_img.shape[0] >= 3:
        # RGB
        image_array = np.transpose(out_img[:3], (1, 2, 0))

        if image_array.dtype != np.uint8:
            valid_pixels = ~mask_pixels_img
            valid_vals = image_array[valid_pixels]

            if valid_vals.size == 0:
                image_array[...] = 0
            else:
                nan_min = np.nanmin(valid_vals)
                nan_max = np.nanmax(valid_vals)
                ptp = nan_max - nan_min

                if ptp == 0:
                    image_array[...] = 0
                else:
                    image_array = ((image_array - nan_min) / ptp * 255).astype(np.uint8)

        image_array[mask_pixels_img] = [0, 0, 0]
        image = Image.fromarray(image_array)

    else:
        # Grayscale
        image_array = out_img[0]

        if image_array.dtype != np.uint8:
            valid_pixels = ~mask_pixels_img
            valid_vals = image_array[valid_pixels]

            if valid_vals.size == 0:
                image_array[...] = 0
            else:
                nan_min = np.nanmin(valid_vals)
                nan_max = np.nanmax(valid_vals)
                ptp = nan_max - nan_min

                if ptp == 0:
                    image_array[...] = 0
                else:
                    image_array = ((image_array - nan_min) / ptp * 255).astype(np.uint8)

        image_array[mask_pixels_img] = 0
        image = Image.fromarray(image_array)

    # Save PNG image (outside all conditionals — image is guaranteed to be defined)
    output_path = os.path.join(output_dir, f'crater_{id}.png')
    image.save(output_path)

    # Compute TRI
    TRI_array = compute_TRI_array(out_image, no_data_value)

    # Clean up (remove zero-only rows and columns)
    TRI_array = np.nan_to_num(TRI_array, nan=0.0)
    TRI_array = TRI_array[~np.all(TRI_array == 0, axis=1)]
    TRI_array = TRI_array[:, ~np.all(TRI_array == 0, axis=0)]

    # Save results
    draw_TRI(TRI_array, id, zone)
    array_to_GeoTIF(TRI_array, coords[0], pixel_size_tb, id, zone, crs)

    TRI_mean_crest = extract_values_from_geotiff(f'results/RG{zone}/TRI/TRI_{id}.tif', max_coord_real)

    return TRI_mean_crest

# These commented lines are used to compute the TRI on an entre DTM
# The process take a lot of time (around 40minutes per DTM
# If you want to run it, delete the 6 ", then run this python file
# Be careful, please put back the 6 " if you want to run main.py
"""
zones = [2, 3, 4, 5, 6, 7, 8]

# Definition of the pixel size and of the vertical precision error for each zone (DTM)
zone_settings = {
    2: {'pixel_sizeb_t': 2, 'precision_error': 0.81},
    3: {'pixel_size_tb': 2, 'precision_error': 0.91},
    4: {'pixel_size_tb': 2, 'precision_error': 0.87},
    5: {'pixel_size_tb': 5, 'precision_error': 2.54},
    6: {'pixel_size_tb': 5, 'precision_error': 2.34},
    7: {'pixel_size_tb': 5, 'precision_error': 2.37},
    8: {'pixel_size_tb': 5, 'precision_error': 1.89}
}

for zone in zones:

    debut = time.time()

    # Load the DTM
    mnt_path = os.path.join('..', 'data', 'RG', 'DTM', f'NAC_DTM_REINER{zone}.tiff')

    params = zone_settings.get(zone)
    pixel_size = params['pixel_size_tb']

    with rasterio.open(mnt_path) as src:
        mnt = src.read(1, masked=False)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        no_data_value = src.nodata

    # Computation of TRI on the entire DTM
    TRI_array = compute_TRI_array(np.expand_dims(mnt, axis=0), no_data_value)

    # NaNs cleaning
    TRI_array = np.nan_to_num(TRI_array, nan=0.0)

    # Save
    id = f'MNT{zone}'      # general id, not the crater id
    zone = 'FULL'   # ou nom du site, par exemple

    # Save the picture in png
    draw_TRI(TRI_array, id, zone)

    # Save in GeoTIFF
    coord_left_down = [transform.c, transform.f - mnt.shape[0] * pixel_size]
    array_to_GeoTIF(TRI_array, coord_left_down, pixel_size, id, zone, crs)

    fin = time.time()
    print(f"Execution time : {fin - debut:.6f} seconds")
"""
