import rasterio
import numpy as np
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

def geotiff_to_shapefile(input_tiff_path, output_shapefile_path):
    # Lire le GeoTIFF
    with rasterio.open(input_tiff_path) as src:
        red = src.read(1)   # canal 1 - Rouge
        green = src.read(2) # canal 2 - Vert
        blue = src.read(3)  # canal 3 - Bleu
        transform = src.transform
        crs = src.crs

        mask = (blue > 200)

        # Convertir le masque en polygones
        shapes_generator = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

        polygons = []
        for geom, value in shapes_generator:
            if value == 1:
                polygons.append(shape(geom))

        # Créer un GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

        # Sauvegarder en shapefile
        gdf.to_file(output_shapefile_path)

# Exemple d'utilisation
geotiff_to_shapefile("data_omat/OMAT_RG.TIF", "results_omat/zones_bleu_200.shp")
