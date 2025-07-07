########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
zones = ['7']

D_on = []
dD_on = []
delta_dD_on_stopar = []
delta_dD_on_hoover = []

D_off = []
dD_off = []
delta_dD_off_stopar = []
delta_dD_off_hoover = []

for zone in zones:
    results_shapefile_path = f'results/RG{zone}/results_geom_08_40m_RG{zone}_v2.shp'
    results = gpd.read_file(results_shapefile_path)

    for row in results.itertuples():
        if row.swirl == 'on-swirl':
            D_on.append(row.moyen_diam)
            dD_on.append(row.ratio_dD)
            delta_dD_on_stopar.append(row.δ_dD_stop)
            delta_dD_on_hoover.append(row.δ_dD_hoov)
        else:
            D_off.append(row.moyen_diam)
            dD_off.append(row.ratio_dD)
            delta_dD_off_stopar.append(row.δ_dD_stop)
            delta_dD_off_hoover.append(row.δ_dD_hoov)

# Convertir en arrays NumPy
D_off = np.array(D_off)
dD_off = np.array(dD_off)
delta_dD_off_stopar = np.array(delta_dD_off_stopar)
delta_dD_off_hoover = np.array(delta_dD_off_hoover)

D_on = np.array(D_on)
dD_on = np.array(dD_on)
delta_dD_on_stopar = np.array(delta_dD_on_stopar)
delta_dD_on_hoover = np.array(delta_dD_on_hoover)

# Trier et tracer pour les "off-swirl"
indices_trie_off = np.argsort(D_off)
D_off_trie = D_off[indices_trie_off]
dD_off_trie = dD_off[indices_trie_off]
delta_dD_off_stopar_trie = delta_dD_off_stopar[indices_trie_off]
delta_dD_off_hoover_trie = delta_dD_off_hoover[indices_trie_off]

plt.figure(figsize=(15, 8))
plt.plot(D_off_trie, dD_off_trie, label='d/D')
plt.errorbar(D_off_trie, dD_off_trie, yerr=delta_dD_off_hoover_trie, fmt='x', capsize=3, label='Uncertainity Hoover')
plt.errorbar(D_off_trie, dD_off_trie, yerr=delta_dD_off_stopar_trie, fmt='x', capsize=3, label='Uncertainity Stopar')
plt.legend()
plt.xlabel('Diameters')
plt.ylabel('d/D')
plt.title('d/D as a function of diameter for off-swirls craters')
plt.grid(True)
plt.show()

# Trier et tracer pour les "on-swirl"
indices_trie_on = np.argsort(D_on)
D_on_trie = D_on[indices_trie_on]
dD_on_trie = dD_on[indices_trie_on]
delta_dD_on_stopar_trie = delta_dD_on_stopar[indices_trie_on]
delta_dD_on_hoover_trie = delta_dD_on_hoover[indices_trie_on]

plt.figure(figsize=(15, 8))
plt.plot(D_on_trie, dD_on_trie, label='d/D')
plt.errorbar(D_on_trie, dD_on_trie, yerr=delta_dD_on_hoover_trie, fmt='x', capsize=3, label='Uncertainity Hoover')
plt.errorbar(D_on_trie, dD_on_trie, yerr=delta_dD_on_stopar_trie, fmt='x', capsize=3, label='Uncertainity Stopar')
plt.legend()
plt.xlabel('Diameters')
plt.ylabel('d/D')
plt.title('d/D as a function of diameter for on-swirls craters')
plt.grid(True)
plt.show()
