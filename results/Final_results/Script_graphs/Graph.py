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

colonnes_numeriques = ['mean_diam', 'ratio_dD', 'δ_dD']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
df['swirl'] = df['swirl'].str.strip()
on_swirl = df[df['swirl'] == 'on-swirl']
off_swirl = df[df['swirl'] == 'off-swirl']

# On-swirl classification
on_swirl_AB = on_swirl[on_swirl['deterior'] == 'AB']
on_swirl_ABB = on_swirl[on_swirl['deterior'] == 'AB - B']
on_swirl_B = on_swirl[on_swirl['deterior'] == 'B']
on_swirl_BBC = on_swirl[on_swirl['deterior'] == 'B - BC']
on_swirl_BC = on_swirl[on_swirl['deterior'] == 'BC']
on_swirl_BCC = on_swirl[on_swirl['deterior'] == 'BC - C']
on_swirl_C = on_swirl[on_swirl['deterior'] == 'C']
on_swirl_Unknown = on_swirl[on_swirl['deterior'] == 'Unknown']

# Off-swirl classification
off_swirl_AB = off_swirl[off_swirl['deterior'] == 'AB']
off_swirl_ABB = off_swirl[off_swirl['deterior'] == 'AB - B']
off_swirl_B = off_swirl[off_swirl['deterior'] == 'B']
off_swirl_BBC = off_swirl[off_swirl['deterior'] == 'B - BC']
off_swirl_BC = off_swirl[off_swirl['deterior'] == 'BC']
off_swirl_BCC = off_swirl[off_swirl['deterior'] == 'BC - C']
off_swirl_C = off_swirl[off_swirl['deterior'] == 'C']
off_swirl_Unknown = off_swirl[off_swirl['deterior'] == 'Unknown']



# Graphique on-swirl
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(on_swirl_AB['mean_diam'], on_swirl_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(on_swirl_ABB['mean_diam'], on_swirl_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(on_swirl_B['mean_diam'], on_swirl_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(on_swirl_BBC['mean_diam'], on_swirl_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(on_swirl_BC['mean_diam'], on_swirl_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(on_swirl_BCC['mean_diam'], on_swirl_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(on_swirl_C['mean_diam'], on_swirl_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(on_swirl_Unknown['mean_diam'], on_swirl_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en fonction du diamètre (on-swirl)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/ratio_dD_vs_mean_diam_on-swirl.png')
plt.close()

# Graphique off-swirl
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(off_swirl_AB['mean_diam'], off_swirl_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(off_swirl_ABB['mean_diam'], off_swirl_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(off_swirl_B['mean_diam'], off_swirl_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(off_swirl_BBC['mean_diam'], off_swirl_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(off_swirl_BC['mean_diam'], off_swirl_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(off_swirl_BCC['mean_diam'], off_swirl_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(off_swirl_C['mean_diam'], off_swirl_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(off_swirl_Unknown['mean_diam'], off_swirl_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en foncton du diamètre (off-swirl)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/ratio_dD_vs_mean_diam_off-swirl.png')
plt.close()

# Graphique on-swirl avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    on_swirl['mean_diam'], on_swirl['ratio_dD'],
    yerr=on_swirl['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    on_swirl['mean_diam'], on_swirl['ratio_dD'],
    s=8, color='blue', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (on-swirl)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/ratio_dD_vs_mean_diam_on-swirl_with_error_bars.png')
plt.close()

# Graphique off-swirl avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    off_swirl['mean_diam'], off_swirl['ratio_dD'],
    yerr=off_swirl['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    off_swirl['mean_diam'], off_swirl['ratio_dD'],
    s=8, color='red', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (off-swirl)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/ratio_dD_vs_mean_diam_off-swirl_with_error_bars.png')
plt.close()

# Ouverture du fichier
df = pd.read_csv('../20250721_RGdD_ALGC_global_results_v1.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['mean_diam', 'ratio_dD', 'δ_dD', 'mean_depth', 'δ_d_1', 'δ_D']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['swirl'] = df['swirl'].str.strip()
on_swirl = df[(df['swirl'] == 'on-swirl') &
              (df['deterior'] != 'C') &
              (df['deterior'] != 'BC - C') &
              (df['deterior'] != 'Unknown')]
off_swirl = df[(df['swirl'] == 'off-swirl') &
               (df['deterior'] != 'C') &
               (df['deterior'] != 'BC - C') &
               (df['deterior'] != 'Unknown')]

plt.figure(figsize=(19, 10))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Données strictement positives pour log-log
on_swirl = on_swirl[(on_swirl['mean_diam'] > 0) & (on_swirl['mean_depth'] > 0)]
off_swirl = off_swirl[(off_swirl['mean_diam'] > 0) & (off_swirl['mean_depth'] > 0)]

# Scatter plots avec barres d'erreur
plt.errorbar(
    on_swirl['mean_diam'], on_swirl['mean_depth'],
    yerr=on_swirl['δ_d_1'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)

plt.errorbar(
    off_swirl['mean_diam'], off_swirl['mean_depth'],
    yerr=off_swirl['δ_d_1'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)

plt.errorbar(
    on_swirl['mean_diam'], on_swirl['mean_depth'],
    xerr=on_swirl['δ_D'],
    fmt='none', ecolor='silver', elinewidth=1, capsize=2,
    zorder=1
)

plt.errorbar(
    off_swirl['mean_diam'], off_swirl['mean_depth'],
    xerr=off_swirl['δ_D'],
    fmt='none', ecolor='silver', elinewidth=1, capsize=2,
    zorder=1
)

plt.scatter(on_swirl['mean_diam'], on_swirl['mean_depth'], s=7.5, color='red', label='on-swirl')
plt.scatter(off_swirl['mean_diam'], off_swirl['mean_depth'], s=7.5, color='blue', label='off-swirl')

# Données combinées pour la régression
all_diam = pd.concat([on_swirl['mean_diam'], off_swirl['mean_diam']])
all_depth = pd.concat([on_swirl['mean_depth'], off_swirl['mean_depth']])

# Régression log-log
log_x = np.log10(all_diam)
log_y = np.log10(all_depth)
coeffs_log = np.polyfit(log_x, log_y, deg=1)
a_log, b_log = coeffs_log

all_mean_diam = pd.concat([on_swirl['mean_diam'], off_swirl['mean_diam']])
x_vals = np.logspace(np.log10(all_mean_diam.min()), np.log10(all_mean_diam.max()), 100)

# Lignes pour ratios constants
ratios = [0.2, 0.1, 0.05]
for r in ratios:
    y_ratio = r * x_vals
    plt.plot(x_vals, y_ratio, linestyle=':', linewidth=1.2, label=f'Ratio = {r}')

# Mise en forme
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.title("Profondeur en fonction du diamètre (échelle log-log)")
plt.xlabel("Diamètre moyen (m)")
plt.ylabel("Profondeur")

# Sauvegarde et affichage
plt.savefig('../Graphs/prof_vs_mean_diam_on-swirl_loglog_deterior.png')
plt.show()
plt.close()



### clementine_color_ratio ###

# Ouverture du fichier
df = pd.read_csv('results_clementine_color_ratio/global_results_clementine_color_ratio.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['mean_diam', 'ratio_dD', 'δ_dD']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
on_clementine_color_ratio = df[df['Join_Count'] >= 1]
off_clementine_color_ratio = df[df['Join_Count']==0]

# On-clementine_color_ratio classification
on_clementine_color_ratio_AB = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'AB']
on_clementine_color_ratio_ABB = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'AB - B']
on_clementine_color_ratio_B = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'B']
on_clementine_color_ratio_BBC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'B - BC']
on_clementine_color_ratio_BC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'BC']
on_clementine_color_ratio_BCC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'BC - C']
on_clementine_color_ratio_C = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'C']
on_clementine_color_ratio_Unknown = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'Unknown']

# Off-clementine_color_ratio classification
off_clementine_color_ratio_AB = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'AB']
off_clementine_color_ratio_ABB = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'AB - B']
off_clementine_color_ratio_B = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'B']
off_clementine_color_ratio_BBC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'B - BC']
off_clementine_color_ratio_BC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'BC']
off_clementine_color_ratio_BCC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'BC - C']
off_clementine_color_ratio_C = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'C']
off_clementine_color_ratio_Unknown = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'Unknown']



# Graphique on-clementine_color_ratio
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(on_clementine_color_ratio_AB['mean_diam'], on_clementine_color_ratio_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(on_clementine_color_ratio_ABB['mean_diam'], on_clementine_color_ratio_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(on_clementine_color_ratio_B['mean_diam'], on_clementine_color_ratio_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(on_clementine_color_ratio_BBC['mean_diam'], on_clementine_color_ratio_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(on_clementine_color_ratio_BC['mean_diam'], on_clementine_color_ratio_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(on_clementine_color_ratio_BCC['mean_diam'], on_clementine_color_ratio_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(on_clementine_color_ratio_C['mean_diam'], on_clementine_color_ratio_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(on_clementine_color_ratio_Unknown['mean_diam'], on_clementine_color_ratio_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en fonction du diamètre (on-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_175/ratio_dD_vs_mean_diam_on-clementine_color_ratio_175.png')
plt.close()

# Graphique off-clementine_color_ratio
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(off_clementine_color_ratio_AB['mean_diam'], off_clementine_color_ratio_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(off_clementine_color_ratio_ABB['mean_diam'], off_clementine_color_ratio_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(off_clementine_color_ratio_B['mean_diam'], off_clementine_color_ratio_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(off_clementine_color_ratio_BBC['mean_diam'], off_clementine_color_ratio_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(off_clementine_color_ratio_BC['mean_diam'], off_clementine_color_ratio_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(off_clementine_color_ratio_BCC['mean_diam'], off_clementine_color_ratio_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(off_clementine_color_ratio_C['mean_diam'], off_clementine_color_ratio_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(off_clementine_color_ratio_Unknown['mean_diam'], off_clementine_color_ratio_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en foncton du diamètre (off-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_175/ratio_dD_vs_mean_diam_off-clementine_color_ratio_175.png')
plt.close()

# Graphique on-clementine_color_ratio avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    on_clementine_color_ratio['mean_diam'], on_clementine_color_ratio['ratio_dD'],
    yerr=on_clementine_color_ratio['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    on_clementine_color_ratio['mean_diam'], on_clementine_color_ratio['ratio_dD'],
    s=8, color='blue', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (on-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_175/ratio_dD_vs_mean_diam_on-clementine_color_ratio_175_with_error_bars.png')
plt.close()

# Graphique off-clementine_color_ratio avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    off_clementine_color_ratio['mean_diam'], off_clementine_color_ratio['ratio_dD'],
    yerr=off_clementine_color_ratio['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    off_clementine_color_ratio['mean_diam'], off_clementine_color_ratio['ratio_dD'],
    s=8, color='red', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (off-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_175/ratio_dD_vs_mean_diam_off-clementine_color_ratio_175_with_error_bars.png')
plt.close()

# clementine_color_ratio 200

# Ouverture du fichier
df = pd.read_csv('results_clementine_color_ratio/global_results_clementine_color_ratio_200.csv', sep=";")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les virgules par des points dans toutes les cellules (le csv travaille avec des , alors que python travaille
# avec des.)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

colonnes_numeriques = ['mean_diam', 'ratio_dD', 'δ_dD']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrer les données
on_clementine_color_ratio = df[df['Join_Count'] >= 1]
off_clementine_color_ratio = df[df['Join_Count'] == 0]

# On-clementine_color_ratio classification
on_clementine_color_ratio_AB = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'AB']
on_clementine_color_ratio_ABB = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'AB - B']
on_clementine_color_ratio_B = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'B']
on_clementine_color_ratio_BBC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'B - BC']
on_clementine_color_ratio_BC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'BC']
on_clementine_color_ratio_BCC = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'BC - C']
on_clementine_color_ratio_C = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'C']
on_clementine_color_ratio_Unknown = on_clementine_color_ratio[on_clementine_color_ratio['deterior'] == 'Unknown']

# Off-clementine_color_ratio classification
off_clementine_color_ratio_AB = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'AB']
off_clementine_color_ratio_ABB = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'AB - B']
off_clementine_color_ratio_B = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'B']
off_clementine_color_ratio_BBC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'B - BC']
off_clementine_color_ratio_BC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'BC']
off_clementine_color_ratio_BCC = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'BC - C']
off_clementine_color_ratio_C = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'C']
off_clementine_color_ratio_Unknown = off_clementine_color_ratio[off_clementine_color_ratio['deterior'] == 'Unknown']



# Graphique on-clementine_color_ratio
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(on_clementine_color_ratio_AB['mean_diam'], on_clementine_color_ratio_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(on_clementine_color_ratio_ABB['mean_diam'], on_clementine_color_ratio_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(on_clementine_color_ratio_B['mean_diam'], on_clementine_color_ratio_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(on_clementine_color_ratio_BBC['mean_diam'], on_clementine_color_ratio_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(on_clementine_color_ratio_BC['mean_diam'], on_clementine_color_ratio_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(on_clementine_color_ratio_BCC['mean_diam'], on_clementine_color_ratio_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(on_clementine_color_ratio_C['mean_diam'], on_clementine_color_ratio_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(on_clementine_color_ratio_Unknown['mean_diam'], on_clementine_color_ratio_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en fonction du diamètre (on-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_200/ratio_dD_vs_mean_diam_on-clementine_color_ratio_200.png')
plt.close()

# Graphique off-clementine_color_ratio
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.scatter(off_clementine_color_ratio_AB['mean_diam'], off_clementine_color_ratio_AB['ratio_dD'], s=7.5, color='hotpink', label='AB')
plt.scatter(off_clementine_color_ratio_ABB['mean_diam'], off_clementine_color_ratio_ABB['ratio_dD'], s=7.5, color='red', label='AB - B')
plt.scatter(off_clementine_color_ratio_B['mean_diam'], off_clementine_color_ratio_B['ratio_dD'], s=7.5, color='orange', label='B')
plt.scatter(off_clementine_color_ratio_BBC['mean_diam'], off_clementine_color_ratio_BBC['ratio_dD'], s=7.5, color='gold', label='B - BC')
plt.scatter(off_clementine_color_ratio_BC['mean_diam'], off_clementine_color_ratio_BC['ratio_dD'], s=7.5, color='green', label='BC')
plt.scatter(off_clementine_color_ratio_BCC['mean_diam'], off_clementine_color_ratio_BCC['ratio_dD'], s=7.5, color='blue', label='BC - C')
plt.scatter(off_clementine_color_ratio_C['mean_diam'], off_clementine_color_ratio_C['ratio_dD'], s=7.5, color='indigo', label='C')
plt.scatter(off_clementine_color_ratio_Unknown['mean_diam'], off_clementine_color_ratio_Unknown['ratio_dD'], s=7.5, color='black', label='Unknown')
plt.legend(loc='lower right')
plt.xlim(0, 2100)
plt.ylim(0, 0.185)
plt.title("Ratio d/D en foncton du diamètre (off-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_200/ratio_dD_vs_mean_diam_off-clementine_color_ratio_200.png')
plt.close()

# Graphique on-clementine_color_ratio avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    on_clementine_color_ratio['mean_diam'], on_clementine_color_ratio['ratio_dD'],
    yerr=on_clementine_color_ratio['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    on_clementine_color_ratio['mean_diam'], on_clementine_color_ratio['ratio_dD'],
    s=8, color='blue', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (on-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_200/ratio_dD_vs_mean_diam_on-clementine_color_ratio_200_with_error_bars.png')
plt.close()

# Graphique off-clementine_color_ratio avec barres d'erreurs
plt.figure(figsize=(19, 10))
plt.grid(True)
plt.errorbar(
    off_clementine_color_ratio['mean_diam'], off_clementine_color_ratio['ratio_dD'],
    yerr=off_clementine_color_ratio['δ_dD'],
    fmt='none', ecolor='grey', elinewidth=1, capsize=2,
    zorder=1
)
plt.scatter(
    off_clementine_color_ratio['mean_diam'], off_clementine_color_ratio['ratio_dD'],
    s=8, color='red', zorder=2
)
plt.xlim(0, 900)
plt.ylim(0, 0.16)
plt.title("Ratio d/D par rapport au diamètre (limité à 900m) avec les barres d'erreurs (off-clementine_color_ratio)")
plt.xlabel("Diamètre moyen")
plt.ylabel("ratio d/D")
plt.savefig('../Graphs/clementine_color_ratio_200/ratio_dD_vs_mean_diam_off-clementine_color_ratio_200_with_error_bars.png')
plt.close()
