import rasterio
from rasterio.mask import mask
import geopandas as gpd

def crop(raster_path, shapefile_path, output_path):

    # Lire le shapefile
    gdf = gpd.read_file(shapefile_path)
    # Assurez-vous que la géométrie est valide
    gdf = gdf[gdf.is_valid]

    # Lire le raster
    with rasterio.open(raster_path) as src:
        # Reprojeter les géométries du shapefile dans le système de coordonnées du raster si nécessaire
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Extraire les géométries sous forme de dictionnaire
        geometries = gdf.geometry.values

        # Découper le raster
        out_image, out_transform = mask(src, geometries, crop=False, nodata=src.nodata)

        # Copier le profil du raster d'origine
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata
        })

    # Sauvegarder le raster découpé
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
