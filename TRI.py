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
    half_size = ray
    coords = [
        [center_x_dl - half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl - half_size],
        [center_x_dl + half_size, center_y_dl + half_size],
        [center_x_dl - half_size, center_y_dl + half_size]
    ]
    polygon = [Polygon(coords)]

    # Clip raster using the polygon
    out_image, out_transform = mask(src, polygon, crop=True)

    # Compute TRI
    TRI_array = compute_TRI_array(out_image, no_data_value)

    # Clean borders and remove all-zeros rows/cols
    TRI_array = np.nan_to_num(TRI_array, nan=0.0)
    TRI_array = TRI_array[~np.all(TRI_array == 0, axis=1)]
    TRI_array = TRI_array[:, ~np.all(TRI_array == 0, axis=0)]

    # Save outputs
    draw_TRI(TRI_array, id, zone)
    array_to_GeoTIF(TRI_array, coords[0], pixel_size_tb, id, zone, crs)

