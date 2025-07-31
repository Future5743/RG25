import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ouverture du fichier
df = pd.read_csv('../20250721_RGdD_ALGC_global_results_v1.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD', 'mean_TRI']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
df['swirl'] = df['swirl'].str.strip()
on_swirl = df[df['swirl'] == 'on-swirl']
off_swirl = df[df['swirl'] == 'off-swirl']

mean_dD_off = round(np.mean(off_swirl['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_swirl['ratio_dD']), 3)

print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_swirl['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_swirl['δ_dD']), 5)}")


# Ouverture du fichier
df = pd.read_csv('results_omat/global_results_omat.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
on_omat = df[df['Join_Count'] >= 1]
off_omat = df[df['Join_Count'] == 0]

print(len(on_omat), len(off_omat))

mean_dD_off = round(np.mean(off_omat['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_omat['ratio_dD']), 3)

print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_omat['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_omat['δ_dD']), 5)}")

# Ouverture du fichier
df = pd.read_csv('results_omat/global_results_omat_200.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
on_omat = df[df['Join_Count'] >= 1]
off_omat = df[df['Join_Count'] == 0]

print(len(on_omat), len(off_omat))

mean_dD_off = round(np.mean(off_omat['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_omat['ratio_dD']), 3)

print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_omat['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_omat['δ_dD']), 5)}")




print('########## Limitation aux cratères de diamètres infereur ou égal à 100m et peu dégradés ##########')

df = pd.read_csv('../20250721_RGdD_ALGC_global_results_v1.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD', 'mean_diam']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
df['swirl'] = df['swirl'].str.strip()

on_swirl = df[
    (df['swirl'] == 'on-swirl') &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]

off_swirl = df[
    (df['swirl'] == 'off-swirl') &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]

df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

mean_dD_off = round(np.mean(off_swirl['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_swirl['ratio_dD']), 3)

print(len(on_swirl), len(off_swirl))
print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_swirl['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_swirl['δ_dD']), 5)}")

# Ouverture du fichier
df = pd.read_csv('results_omat/global_results_omat.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD', 'mean_diam']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données

on_omat = df[
    (df['Join_Count'] >= 1) &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]

off_omat = df[
    (df['Join_Count'] == 0) &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]

df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
print(len(on_omat), len(off_omat))

mean_dD_off = round(np.mean(off_omat['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_omat['ratio_dD']), 3)

print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_omat['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_omat['δ_dD']), 5)}")

# Ouverture du fichier
df = pd.read_csv('results_omat/global_results_omat_200.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['ratio_dD', 'δ_dD', 'mean_diam']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données

on_omat = df[
    (df['Join_Count'] >= 1) &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]

off_omat = df[
    (df['Join_Count'] == 0) &
    (df['mean_diam'] <= 400) &
    (df['mean_diam'] > 200) &
    (df['deterior'] != 'Unknown') &
    (df['deterior'] != 'C') &
    (df['deterior'] != 'BC - C')
]
print(len(on_omat), len(off_omat))

mean_dD_off = round(np.mean(off_omat['ratio_dD']), 3)
mean_dD_on = round(np.mean(on_omat['ratio_dD']), 3)

print(f"Le d/D moyen des cratères en dehors du tourbillon est {mean_dD_off}±{round(np.mean(off_omat['δ_dD']), 5)}")
print(f"Le d/D moyen des cratères sur le tourbillon est {mean_dD_on}±{round(np.mean(on_omat['δ_dD']), 5)}")
