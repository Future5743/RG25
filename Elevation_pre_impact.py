import numpy as np
from osgeo import gdal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt


def comble_trous_dtm(input_raster_path, output_raster_path, method='cubic', smooth=True, sigma=1):
    """
    Comble les trous dans un DTM par interpolation + génère une carte de fiabilité.

    :param input_raster_path: Chemin du raster d'entrée
    :param output_raster_path: Chemin du raster corrigé
    :param method: Méthode d'interpolation ('linear', 'nearest', 'cubic')
    :param smooth: Lissage après interpolation (bool)
    :param sigma: Paramètre du filtre gaussien
    """
    ds = gdal.Open(input_raster_path)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    array = band.ReadAsArray()
    rows, cols = array.shape

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    # Coordonnées raster (géoréférencées)
    x_coords = np.arange(cols) * gt[1] + gt[0]
    y_coords = np.arange(rows) * gt[5] + gt[3]
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Masque des données valides
    valid_mask = (array != nodata) & (~np.isnan(array))
    array = np.where(valid_mask, array, np.nan)

    # Interpolation principale
    interp_values = griddata(
        (xx[valid_mask], yy[valid_mask]),
        array[valid_mask],
        (xx, yy),
        method=method
    )

    # Remplacer uniquement les trous
    filled_array = np.where(valid_mask, array, interp_values)

    # Appliquer un lissage uniquement sur les pixels interpolés
    if smooth:
        smoothed = gaussian_filter(filled_array, sigma=sigma)
        filled_array = np.where(valid_mask, array, smoothed)

    # Générer la carte de fiabilité
    distance_map = distance_transform_edt(~valid_mask)
    max_dist = np.max(distance_map)
    fiabilite_array = 1 - np.clip(distance_map / max_dist, 0, 1)
    fiabilite_array = np.where(valid_mask, 1.0, fiabilite_array)

    # 🔒 Clipping final pour s'assurer des bonnes dimensions
    filled_array = filled_array[:rows, :cols]
    fiabilite_array = fiabilite_array[:rows, :cols]

    # Sauvegarde du raster interpolé
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster_path, cols, rows, 1, band.DataType)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(filled_array)
    out_band.SetNoDataValue(nodata)
    out_band.FlushCache()
    out_ds = None

    print(f"✅ Raster comblé sauvegardé dans : {output_raster_path}")

    # Sauvegarde de la carte de fiabilité
    fiabilite_path = output_raster_path.replace(".TIF", "_fiabilite.TIF")
    fiab_ds = driver.Create(fiabilite_path, cols, rows, 1, gdal.GDT_Float32)
    fiab_ds.SetGeoTransform(gt)
    fiab_ds.SetProjection(proj)
    fiab_ds.GetRasterBand(1).WriteArray(fiabilite_array)
    fiab_ds.GetRasterBand(1).SetNoDataValue(0)
    fiab_ds.FlushCache()
    fiab_ds = None

    print(f"📊 Carte de fiabilité sauvegardée dans : {fiabilite_path}")


# Exemple d'appel
comble_trous_dtm(
    input_raster_path="../data/RG/RG2_clip_02.TIF",
    output_raster_path="../data/RG/RG2_interpolation_robuste_02.TIF",
    method='linear',
    smooth=True,
    sigma=1
)
