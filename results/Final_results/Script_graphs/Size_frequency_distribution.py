import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Charger le shapefile ===
shapefile_path = "../20250721_RGdD_ALGC_global_results_v1.shp"
gdf = gpd.read_file(shapefile_path)

# Vérification des colonnes
if 'swirl' not in gdf.columns or 'mean_diam' not in gdf.columns:
    raise ValueError("Le shapefile doit contenir les colonnes 'swirl' et 'mean_diam'.")

# === 2. Filtrer les données ===
off_swirl = gdf[gdf['swirl'] == 'off-swirl']
on_swirl = gdf[gdf['swirl'] == 'on-swirl']

# === 3. Définir les bornes des classes ===
all_diams = gdf['mean_diam'].dropna()
min_diam = int(np.floor(all_diams.min() / 10.0) * 10)
max_diam = int(np.ceil(all_diams.max() / 10.0) * 10)
bins = np.arange(min_diam, max_diam + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2  # centres des classes

# === 4. Calculer les effectifs par classe ===
off_counts, _ = np.histogram(off_swirl['mean_diam'].dropna(), bins=bins)
on_counts, _ = np.histogram(on_swirl['mean_diam'].dropna(), bins=bins)

# === 5. Tracer le graphique classique en échelle log-log ===
plt.figure(figsize=(10, 6))
plt.scatter(bin_centers, off_counts / 1532854155.9729, label='off-swirl', color='blue')
plt.scatter(bin_centers, on_counts / 1168310482.72885, label='on-swirl', color='orange')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Densité de cratères (nb/m²)')
plt.title('Distribution taille-fréquence des cratères (points uniquement)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("../Graphs/Size_frequency_on_and_off_swirl.png")
plt.show()

# Off-swirl seul
plt.figure(figsize=(10, 6))
plt.scatter(bin_centers, off_counts / 1532854155.9729, label='off-swirl', color='blue')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Densité de cratères (nb/m²)')
plt.title('Distribution taille-fréquence des cratères (hors tourbillon)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("../Graphs/Size_frequency_off_swirl.png")
plt.show()

# On-swirl seul
plt.figure(figsize=(10, 6))
plt.scatter(bin_centers, on_counts / 1168310482.72885, label='on-swirl', color='orange')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Densité de cratères (nb/m²)')
plt.title('Distribution taille-fréquence des cratères (sur tourbillon)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("../Graphs/Size_frequency_on_swirl.png")
plt.show()


### clementine_color_ratio ###
shapefile_path = "results_clementine_color_ratio/global_results_clementine_color_ratio.shp"
gdf = gpd.read_file(shapefile_path)


# === 2. Filtrer les données ===
off_clementine_color_ratio = gdf[gdf['Join_Count'] == 0]
on_clementine_color_ratio = gdf[gdf['Join_Count'] >= 1]

# === 3. Définir les bornes des classes ===
all_diams = gdf['mean_diam'].dropna()
min_diam = int(np.floor(all_diams.min() / 10.0) * 10)
max_diam = int(np.ceil(all_diams.max() / 10.0) * 10)
bins = np.arange(min_diam, max_diam + 10, 10)

# === 4. Tracer les histogrammes ===
plt.figure(figsize=(10, 6))

plt.hist(off_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue inferieure à 175', color='blue', edgecolor='black')
plt.hist(on_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue supérieure à 175', color='orange', edgecolor='black')

plt.axvline(off_clementine_color_ratio['mean_diam'].dropna().mean(), color='blue', linestyle='dashed', linewidth=1.5)
plt.axvline(on_clementine_color_ratio['mean_diam'].dropna().mean(), color='orange', linestyle='dashed', linewidth=1.5)

plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères en \nfonction de leur valeur sur le canal bleu de clementine_color_ratio (inferieure ou supérieure à 175)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_175/Size_distribution_on_and_off_clementine_color_ratio_175.png")
plt.show()

# Off-swirl seulement
plt.figure(figsize=(10, 6))
plt.hist(off_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue inferieure à 175', color='blue', edgecolor='black')
plt.axvline(off_clementine_color_ratio['mean_diam'].dropna().mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères ayant une valeur de canal bleu \nde clementine_color_ratio inferieure à 175")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_175/Size_distribution_off_clementine_color_ratio_175.png")
plt.show()

# On-swirl seulement
plt.figure(figsize=(10, 6))
plt.hist(on_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue supérieure à 175', color='orange', edgecolor='black')
plt.axvline(on_clementine_color_ratio['mean_diam'].dropna().mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères ayant une valeur de canal bleu \nde clementine_color_ratio supérieure à 175")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_175/Size_distribution_on_clementine_color_ratio_175.png")
plt.show()

# clementine_color_ratio 200
shapefile_path = "results_clementine_color_ratio/global_results_clementine_color_ratio_200.shp"
gdf = gpd.read_file(shapefile_path)

# Vérifier que les colonnes nécessaires existent

# === 2. Filtrer les données ===
off_clementine_color_ratio = gdf[gdf['Join_Count'] == 0]
on_clementine_color_ratio = gdf[gdf['Join_Count'] >= 1]

# === 3. Définir les bornes des classes ===
all_diams = gdf['mean_diam'].dropna()
min_diam = int(np.floor(all_diams.min() / 10.0) * 10)
max_diam = int(np.ceil(all_diams.max() / 10.0) * 10)
bins = np.arange(min_diam, max_diam + 10, 10)

# === 4. Tracer les histogrammes ===
plt.figure(figsize=(10, 6))

plt.hist(off_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue inferieure à 200', color='blue', edgecolor='black')
plt.hist(on_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue supérieure à 200', color='orange', edgecolor='black')

plt.axvline(off_clementine_color_ratio['mean_diam'].dropna().mean(), color='blue', linestyle='dashed', linewidth=1.5)
plt.axvline(on_clementine_color_ratio['mean_diam'].dropna().mean(), color='orange', linestyle='dashed', linewidth=1.5)

plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères en \nfonction de leur valeur sur le canal bleu de clementine_color_ratio (inferieure ou supérieure à 200)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_200/Size_distribution_on_and_off_clementine_color_ratio_200.png")
plt.show()

# Off-swirl seulement
plt.figure(figsize=(10, 6))
plt.hist(off_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue inferieure à 200', color='blue', edgecolor='black')
plt.axvline(off_clementine_color_ratio['mean_diam'].dropna().mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères ayant une valeur de canal bleu \nde clementine_color_ratio inferieure à 200")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_200/Size_distribution_off_clementine_color_ratio_200.png")
plt.show()

# On-swirl seulement
plt.figure(figsize=(10, 6))
plt.hist(on_clementine_color_ratio['mean_diam'].dropna(), bins=bins, alpha=0.6, label='Valeur bleue supérieure à 200', color='orange', edgecolor='black')
plt.axvline(on_clementine_color_ratio['mean_diam'].dropna().mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.xlabel('Diamètre moyen (m)')
plt.ylabel('Nombre de cratère')
plt.title("Distribution du diamètre des cratères ayant une valeur de canal bleu \nde clementine_color_ratio supérieure à 200")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/clementine_color_ratio_200/Size_distribution_on_clementine_color_ratio_200.png")
plt.show()
