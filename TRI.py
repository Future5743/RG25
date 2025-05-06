######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################
import matplotlib.pyplot as plt

import os

import rasterio

from rasterio.crs import CRS

from rasterio.mask import mask

from rasterio.transform import from_origin

import numpy as np

from shapely.geometry import Polygon

from shapely.geometry.polygon import orient

######################################################################################################################################################################################
#################################################################################### LIST CREATION ###################################################################################
######################################################################################################################################################################################
def draw_TRI(TRI, id, zone):
    fig, ax = plt.subplots()

    # Afficher les données avec la colormap viridis inversée
    cax = ax.imshow(TRI, cmap='viridis_r', vmin=-0.25, vmax=0.25)

    # Ajouter une barre de couleur
    fig.colorbar(cax)
    plt.title("Indice TRI sur le cratère " + str(id) + " de la zone RG" + zone)

    if not os.path.exists("results/RG" + zone + "/TRI"):
        os.makedirs("results/RG" + zone + "/TRI")

    plt.savefig("results/RG" + zone + "/TRI/TRI_" + str(id) + ".png")
    plt.close()


def array_to_GeoTIF(TRI, coord_left_down, pixel_size_tb, id, zone, crs):
    upper_left_x, upper_left_y = coord_left_down

    transform = from_origin(upper_left_x, upper_left_y, pixel_size_tb, pixel_size_tb)

    with rasterio.open(
            'results/RG' + zone + '/TRI/TRI_' + str(id) + '.tif',
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

def TRI (center_x_dl, center_y_dl, ray, src, no_data_value, pixel_size_tb, id, zone, crs) :
    coord1 = [center_x_dl - ray, center_y_dl - ray]
    coord2 = [center_x_dl + ray, center_y_dl - ray]
    coord3 = [center_x_dl + ray, center_y_dl + ray]
    coord4 = [center_x_dl - ray, center_y_dl + ray]

    emporte_piece = [Polygon([coord1, coord2, coord3, coord4, coord1])]

    # Découper le raster en utilisant le polygone
    out_image, out_transform = mask(src, emporte_piece, crop=True)

    # Ignorer les valeurs "no data"
    masked_image_square = np.ma.masked_equal(out_image, no_data_value)

    DTM_mean = []
    DTM_range = []

    for y in range(1, masked_image_square.shape[1] - 1):

        row_mean = []
        row_range = []

        for x in range(1, masked_image_square.shape[2] - 1):
            window = [masked_image_square[:, y + 1, x - 1],
                      masked_image_square[:, y + 1, x],
                      masked_image_square[:, y + 1, x + 1],
                      masked_image_square[:, y, x - 1],
                      masked_image_square[:, y, x + 1],
                      masked_image_square[:, y - 1, x - 1],
                      masked_image_square[:, y - 1, x],
                      masked_image_square[:, y - 1, x + 1]]

            row_mean.append(np.mean(window))

            row_range.append(np.std(window))

        DTM_mean.append(row_mean)
        DTM_range.append(row_range)

    DTM_mean = np.array(DTM_mean)

    DTM_range = np.array(DTM_range)

    TRI = np.round((masked_image_square[0][1:-1, 1:-1] - DTM_mean) / DTM_range, decimals=3)

    index = np.where(np.isnan(TRI) != False)
    TRI[index] = 0

    TRI = TRI[~np.all(TRI == 0, axis=1)]
    TRI = TRI[:, ~np.all(TRI == 0, axis=0)]

    draw_TRI(TRI, id, zone)

    array_to_GeoTIF(TRI, coord4, pixel_size_tb, id, zone, crs)

