######################################################################################################################################################################################
#################################################################################### IMPORTATIONS ####################################################################################
######################################################################################################################################################################################

import matplotlib.pyplot as plt

import os

import math

import numpy as np

######################################################################################################################################################################################
######################################################################################## CODE ########################################################################################
######################################################################################################################################################################################

def calcul_distance(pos1, pos2, pixel_size_tb=2):
    pixel_dist_tb = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_in_meters_tb = pixel_dist_tb * pixel_size_tb

    return distance_in_meters_tb

def profils_topo(profils, demi_profils_coords_relatives, pixel_size_tb, id, zone, swirl_on_or_off) :
    all_profiles = []
    min_X = [0] * 1000  # Trouver une variable plus exacte que 1000

    ### Creation des 18 profils du cratere
    for i in range(int(len(profils) / 2)):

        X = []

        demi_profil = profils[i + 18]

        reversed_demi_profil = list(reversed(profils[i]))[:-1]

        limit_profil = len(reversed_demi_profil)

        demi_profils_coords_relatives[i][0] = list(reversed(demi_profils_coords_relatives[i][0]))[:-1]
        demi_profils_coords_relatives[i][1] = list(reversed(demi_profils_coords_relatives[i][1]))[:-1]

        demi_profils_coords_relatives[i + 18][0] = list(demi_profils_coords_relatives[i + 18][0])
        demi_profils_coords_relatives[i + 18][1] = list(demi_profils_coords_relatives[i + 18][1])

        profils_coords_relatives_rr = demi_profils_coords_relatives[i][0] + demi_profils_coords_relatives[i + 18][0]
        profils_coords_relatives_cc = demi_profils_coords_relatives[i][1] + demi_profils_coords_relatives[i + 18][1]

        for pixel in range(len(profils_coords_relatives_rr)):
            dist = calcul_distance([profils_coords_relatives_rr[0], profils_coords_relatives_cc[0]],
                                   [profils_coords_relatives_rr[pixel], profils_coords_relatives_cc[pixel]], pixel_size_tb)
            X.append(dist)

        if len(X) < len(min_X):
            min_X = X

        # Gestion du profil
        profil = reversed_demi_profil + demi_profil

        profil = [x if isinstance(x, np.float32) else np.nan for x in profil]

        all_profiles.append(profil)

        # Création de la figure
        plt.figure(figsize=(40, 15))
        plt.plot(X, profil, marker='x')
        plt.xlabel("Distance (m)")
        plt.ylabel("Altitude")
        plt.title("Profil topographique pour les angles " + str(i * 10) + " et " + str((i + 18) * 10))
        plt.grid(True)

        # Gestion des dossiers
        if swirl_on_or_off == 'on-swirl':
            path = "results/RG" + zone + "/profils/on_swirl/" + str(id)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = "results/RG" + zone + "/profils/off_swirl/" + str(id)
            if not os.path.exists(path):
                os.makedirs(path)

        plt.savefig(path + "/Profil_" + str(i * 10) + "_" + str((i + 18) * 10) + ".png")
        plt.close()

    ### Adaptation des profils pour le moyennage futur
    for profil_individuel in all_profiles:
        if len(profil_individuel) > len(min_X):
            excedant = len(profil_individuel) - len(min_X)

            if excedant % 2 == 0:
                profil_individuel = profil_individuel[int(excedant / 2): - int(excedant / 2)]
            else:

                if len(profil_individuel) / 2 < limit_profil:
                    profil_individuel = profil_individuel[math.ceil(excedant / 2): - int(excedant / 2)]
                else:
                    profil_individuel = profil_individuel[int(excedant / 2): - math.ceil(excedant / 2)]

    ### Moyennage des profils
    profil_moyen = []
    for x in range(len(min_X)):
        colonne_i = [sous_liste[x] for sous_liste in all_profiles]
        profil_moyen.append(np.mean(colonne_i))

    plt.figure(figsize=(40, 15))
    plt.plot(min_X, profil_moyen, marker='x')
    plt.xlabel("Distance (m)")
    plt.ylabel("Altitude")
    plt.title("Moyenne des profils topographiques")
    plt.grid(True)
    plt.savefig(path + "/Profil_moyen.png")
    plt.close()