import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import fiona


def raster_to_valid_extent_shapefile(raster_path, shapefile_path):
    '''
    This function create a shapefile with the same extent as the raster.
    :param raster_path: str         -- Is the path where the raster will be stored
    :param shapefile_path:str       -- Is the path where the shapefile will be stored
    :return: No data
    '''

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        nodata = src.nodata

        # Création d'un masque binaire : True pour les données valides
        if nodata is not None:
            mask = band != nodata
        else:
            # Si aucune valeur NoData n'est définie, on considère toutes les valeurs > 0 comme valides (optionnel)
            mask = band > 0

        # Convertir le masque en uint8 (0/1)
        mask_uint8 = mask.astype(np.uint8)

        # Extraire les formes où mask == 1
        valid_shapes = shapes(mask_uint8, mask=mask, transform=transform)

        # Construire les polygones
        geoms = [shape(geom) for geom, val in valid_shapes if val == 1]

        if not geoms:
            print("Aucune zone de données valides trouvée dans le raster.")
            return

        # Fusionner tous les polygones
        merged_geom = unary_union(geoms)

        # Écriture du shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'}
        }

        with fiona.open(
            shapefile_path,
            'w',
            driver='ESRI Shapefile',
            crs=src.crs.to_dict(),
            schema=schema
        ) as shp:
            shp.write({
                'geometry': mapping(merged_geom),
                'properties': {'id': 1}
            })

    print(f"Shapefile enregistré : {shapefile_path}")