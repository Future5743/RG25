######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################

zones = ['2', '7']

D_on = []
dD_on = []
delta_dD_on = []

D_off = []
dD_off = []
delta_dD_off = []

for zone in zones :
    results_shapefile_path = 'results/RG' + zone + '/results_geom_08_40m_RG' + zone + '_v2.shp'
    results = gpd.read_file(results_shapefile_path)

    for row in results.itertuples():
        if row.swirl == 'on-swirl':
            D_on.append(row.moyen_diam)
            dD_on.append(row.ratio_dD)
            delta_dD_on.append(row.incer_dD)
        else:
            D_off.append(row.moyen_diam)
            dD_off.append(row.ratio_dD)
            delta_dD_off.append(row.incer_dD)

# Convertir en arrays NumPy
D_off = np.array(D_off)
dD_off = np.array(dD_off)
delta_dD_off = np.array(delta_dD_off)

D_on = np.array(D_on)
dD_on = np.array(dD_on)
delta_dD_on = np.array(delta_dD_on)

# Trier et tracer pour les "off-swirl"
indices_trie_off = np.argsort(D_off)
D_off_trie = D_off[indices_trie_off]
dD_off_trie = dD_off[indices_trie_off]
delta_dD_off_trie = delta_dD_off[indices_trie_off]

plt.figure(figsize=(15,8))
plt.plot(D_off_trie, dD_off_trie, label='x')
plt.errorbar(D_off_trie, dD_off_trie, yerr=delta_dD_off_trie, fmt='x', capsize=3, label='off-swirl')
plt.xlabel('Diameters')
plt.ylabel('d/D')
plt.title('d/D as a function of diameter for off-swirls craters')
plt.grid(True)
plt.show()

# Trier et tracer pour les "on-swirl"
indices_trie_on = np.argsort(D_on)
D_on_trie = D_on[indices_trie_on]
dD_on_trie = dD_on[indices_trie_on]
delta_dD_on_trie = delta_dD_on[indices_trie_on]

plt.figure(figsize=(15,8))
plt.plot(D_on_trie, dD_on_trie, label='x')
plt.errorbar(D_on_trie, dD_on_trie, yerr=delta_dD_on_trie, fmt='x', color='orange', capsize=3, label='on-swirl')
plt.xlabel('Diameters')
plt.ylabel('d/D')
plt.title('d/D as a function of diameter for on-swirls craters')
plt.grid(True)
plt.show()






