import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
from Emprise_raster import raster_to_valid_extent_shapefile
from Crop_raster import crop


def idw_interpolation(xy, values, xi, power=2, k=12):
    """
    Interpolation par IDW (Inverse Distance Weighting)
    :param xy: Coordonnées des points connus (N x 2)
    :param values: Valeurs connues (N,)
    :param xi: Coordonnées où interpoler (M x 2)
    :param power: Puissance de la pondération (1 = linéaire, 2 = quadratique)
    :param k: Nombre de voisins à considérer
    :return: Valeurs interpolées
    """
    tree = cKDTree(xy)
    distances, idx = tree.query(xi, k=k, p=2)

    weights = 1 / (distances ** power + 1e-12)  # éviter division par zéro
    weighted_values = np.sum(weights * values[idx], axis=1)
    weights_sum = np.sum(weights, axis=1)
    return weighted_values / weights_sum


def interpolation_dtm(input_raster_path, output_raster_path, method='linear', smooth=True, sigma=1, idw_power=2):
    ds = gdal.Open(input_raster_path)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    array = band.ReadAsArray()
    rows, cols = array.shape

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    x_coords = np.arange(cols) * gt[1] + gt[0]
    y_coords = np.arange(rows) * gt[5] + gt[3]
    xx, yy = np.meshgrid(x_coords, y_coords)

    valid_mask = (array != nodata) & (~np.isnan(array))
    array = np.where(valid_mask, array, np.nan)

    xy_known = np.column_stack((xx[valid_mask], yy[valid_mask]))
    z_known = array[valid_mask]
    xy_interp = np.column_stack((xx[~valid_mask], yy[~valid_mask]))

    if method == 'idw':
        print("⚙️ Interpolation par IDW...")
        interp_values = idw_interpolation(xy_known, z_known, xy_interp, power=idw_power)
        interp_full = array.copy()
        interp_full[~valid_mask] = interp_values
    else:
        from scipy.interpolate import griddata
        print(f"⚙️ Interpolation par {method}...")
        interp_full = griddata(
            (xx[valid_mask], yy[valid_mask]),
            z_known,
            (xx, yy),
            method=method
        )

    filled_array = np.where(valid_mask, array, interp_full)

    if smooth:
        smoothed = gaussian_filter(filled_array, sigma=sigma)
        filled_array = np.where(valid_mask, array, smoothed)

    distance_map = distance_transform_edt(~valid_mask)
    max_dist = np.max(distance_map)
    fiabilite_array = 1 - np.clip(distance_map / max_dist, 0, 1)
    fiabilite_array = np.where(valid_mask, 1.0, fiabilite_array)

    filled_array = filled_array[:rows, :cols]
    fiabilite_array = fiabilite_array[:rows, :cols]

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

    fiabilite_path = output_raster_path.replace(".TIF", "_fiabilite.TIF")
    fiab_ds = driver.Create(fiabilite_path, cols, rows, 1, gdal.GDT_Float32)
    fiab_ds.SetGeoTransform(gt)
    fiab_ds.SetProjection(proj)
    fiab_ds.GetRasterBand(1).WriteArray(fiabilite_array)
    fiab_ds.GetRasterBand(1).SetNoDataValue(0)
    fiab_ds.FlushCache()
    fiab_ds = None

    print(f"📊 Carte de fiabilité sauvegardée dans : {fiabilite_path}")

zones = [2,3,4,5,6,7,8]

for zone in zones:

    interpolation_dtm(
        input_raster_path=f"../../data/RG/DTM_crop/RG{zone}_clip_02.TIF",
        output_raster_path=f"../../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation.TIF",
        method='linear',
        smooth=True,
        sigma=1
    )

    crop(raster_path=f"../../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation.TIF",
        shapefile_path=f"../../data/RG/DTM/Emprises/emprise_RG{zone}.shp",
        output_path=f"../../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation_crop.TIF")

    crop(raster_path=f"../../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation_fiabilite.TIF",
        shapefile_path=f"../../data/RG/DTM/Emprises/emprise_RG{zone}.shp",
        output_path=f"../../data/RG/DTM_interpolate/Linear/RG{zone}_linear_interpolation_fiabilite_crop.TIF")



