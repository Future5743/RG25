########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter
import geopandas as gpd
from PIL import Image


########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
def draw_TRI(TRI, id, zone):
    '''
    This function draw and save the TRI index as a color image.

    Entries:
        TRI: array          -- Contains the values of the TRI for the studied crater
        id: int             -- ID of the studied crater
        zone: str           -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)

    Exit data:
        no exit data
    '''

    os.makedirs(f'results/RG{zone}/TRI', exist_ok=True)

    plt.figure()
    cax = plt.imshow(TRI, cmap='viridis_r', vmin=-0.25, vmax=0.25)
    plt.colorbar(cax)
    plt.title(f'Indice TRI sur le cratère {id} de la zone RG{zone}')
    plt.savefig(f'results/RG{zone}/TRI/TRI_{id}.png')
    plt.close()



def array_to_GeoTIF(TRI, coord_left_down, pixel_size_tb, id, zone, crs):
    '''
    This function export the TRI array as a GeoTIFF.

    Entries:
        TRI: array                      -- Contains the values of the TRI for the studied crater
        coord_left_down: list           -- Coordinate of the pixel that is at the bottom left of the TRI matrix
        pixel_size_tb: int              -- Size of the pixel on the terrain
        id: int                         -- ID of the studied crater
        zone: str                       -- Indicate the crater's zone of study (can be 1, 2, 3, 4, 5, 6 or 7)
        crs: str                        -- Is the crs of teh studied area

    Exit data:
        no exit data
    '''

    transform = from_origin(coord_left_down[0], coord_left_down[1] + pixel_size_tb * TRI.shape[0], pixel_size_tb,
                            pixel_size_tb)

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
    """Compute the TRI array using mean and std of the 8-neighbor window."""
    # Mask no-data
    image = np.where(image == no_data_value, np.nan, image)

    # Define function to compute TRI value at each pixel
    def tri_func(window):
        center = window[4]
        neighbors = np.delete(window, 4)
        mean_val = np.nanmean(neighbors)
        std_val = np.nanstd(neighbors)
        if np.isnan(center) or np.isnan(mean_val) or std_val == 0:
            return 0
        return (center - mean_val) / std_val

    # Apply generic filter (3x3 window, flattened)
    return generic_filter(image[0], tri_func, size=3, mode='constant', cval=np.nan)



def TRI(center_x_dl, center_y_dl, ray, src, no_data_value, pixel_size_tb, id, zone, crs):
    """Main function to compute TRI index for a crater area."""
    # Define square polygon around center
    # Définir le polygone à découper (carré centré)
    half_size = ray
    coords = [
        [center_x_dl - half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl + half_size],
        [center_x_dl - half_size, center_y_dl + half_size]
    ]
    polygon = [Polygon(coords)]

    # Découper le raster
    out_image, out_transform = mask(src, polygon, crop=True)
    out_meta = src.meta
    nodata_value = out_meta.get("nodata", None)

    # Créer le masque des pixels à rendre noirs (nodata ou NaN)
    mask_pixels = np.zeros_like(out_image[0], dtype=bool)

    if nodata_value is not None:
        mask_pixels |= np.any(out_image == nodata_value, axis=0)

    if np.issubdtype(out_image.dtype, np.floating):
        mask_pixels |= np.any(np.isnan(out_image), axis=0)

    # Créer dossier de sortie si nécessaire
    output_dir = f'results/RG{zone}/crater_img'
    os.makedirs(output_dir, exist_ok=True)

    # Traitement de l'image
    if out_image.shape[0] >= 3:
        # RGB (au moins 3 bandes)
        image_array = np.transpose(out_image[:3], (1, 2, 0))

        if image_array.dtype != np.uint8:
            valid_pixels = ~mask_pixels
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

        image_array[mask_pixels] = [0, 0, 0]
        image = Image.fromarray(image_array)

    else:
        # Niveaux de gris
        image_array = out_image[0]

        if image_array.dtype != np.uint8:
            valid_pixels = ~mask_pixels
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

        image_array[mask_pixels] = 0
        image = Image.fromarray(image_array)

    # Enregistrer l'image PNG
    output_path = os.path.join(output_dir, f'crater_{id}.png')
    image.save(output_path)
    print(f"✅ Image enregistrée")

    # Compute TRI
    TRI_array = compute_TRI_array(out_image, no_data_value)

    # Clean borders and remove all-zeros rows/cols
    TRI_array = np.nan_to_num(TRI_array, nan=0.0)
    TRI_array = TRI_array[~np.all(TRI_array == 0, axis=1)]
    TRI_array = TRI_array[:, ~np.all(TRI_array == 0, axis=0)]

    # Save outputs
    draw_TRI(TRI_array, id, zone)
    array_to_GeoTIF(TRI_array, coords[0], pixel_size_tb, id, zone, crs)

    print(f"✅ TRI effectué")

