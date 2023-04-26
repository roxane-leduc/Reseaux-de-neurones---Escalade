import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pandas import DataFrame
from scipy.io import loadmat

# ====================================================================
def load_datasets():
    rows = []
    for filename in glob('data/G*.mat'):
        # print(f'Lecture de {filename}')
        monDict = {
            "personne": filename[8:11], 
            "s": filename[12:14],
            "v": filename[15:17],
            "e": filename[18:20],
            "date": filename[21:29],
            }
        monDict.update(loadmat(filename))
        rows.append(monDict)
        # print(loadmat(filename))
    return DataFrame(data=rows)

def clean_datasets(df):
    df.drop(columns=['__header__', '__version__', '__globals__'], inplace=True)
    df.Jerk_rot = df.Jerk_rot.apply(lambda v: v[0,0])
    df.Jerk_pos = df.Jerk_pos.apply(lambda v: v[0,0])
    for fieldname in ['Reduced_state', 'Neck_local_var', 'Hip_local_var', 'Hip_roll', 'Right_hand', 'Neck_roll', 'Left_foot', 'Y', 'X', 'Roll_correl', 'Left_hand', 'Afford_count', 'Right_foot']:
        # Conversion matrice -> vecteur colonne
        df[fieldname] = df[fieldname].apply(lambda v: v.reshape(-1))
    return df

def printTrajectoires(df):
    for idx, row in df.iterrows():
        v, s, e = row.v, row.s, row.e
        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        X, Y = row.X, row.Y
        ax.plot(X, Y, ':k')
        plt.savefig(f'./img/traj_{idx}_{s}_{v}_{e}.pdf', bbox_inches='tight')
        # plt.show()
        # break
        plt.close(fig)

# ====================================================================
# IMPORT DU JEU DE DONNEES
# ====================================================================

df = load_datasets()
dfOrig = df.copy(deep=True)
df = clean_datasets(df)

# print(df.columns.values)
# print(df.Hip_roll.apply(lambda v: v.max()).max())
# df.Hip_roll.apply(lambda v: v.min()).min()

# ====================================================================
# ANALYSE DU JEU DE DONNEES
# ====================================================================

# Boite a moustache sur les donnees jerk
# plt.boxplot(df.Jerk_rot, widths=0.6, showmeans=True, showfliers=False)
# plt.title("Donnees Jerk_rot")
# plt.show()

# plt.boxplot(df.Jerk_pos, widths=0.6, showmeans=True, showfliers=False)
# plt.title("Donnees Jerk_pos")
# plt.show()

# Analyse des moyennes sur les valeurs Hip_roll
mu=[]; std=[]
for i in range(104):
    m, s = np.mean(df.Hip_roll[i], axis=0), np.std(df.Hip_roll[i], axis=0)
    mu.append(m), std.append(s)

muu = sum(mu)/len(mu)
print("Moyenne de toutes les valeurs Hip_roll:", muu)

# Trace de Y en fonction de X
printTrajectoires(df)

# Correl entre les jerk
df_jerk = df.groupby(['personne']).mean()
df_jerk_sort = df_jerk.sort_values('Jerk_pos', ascending = False)
print(df_jerk_sort)
print(np.corrcoef(df.Jerk_rot,df.Jerk_pos))

# Comparer les tailles respectives des colonnes -> 2 types de donnees
def comparaison(df, col1, col2):
    _df = df.copy(deep=True)
    _df['comparaison'] = list(zip(_df[col1], _df[col2]))
    _df.comparaison = _df.comparaison.apply(lambda t: len(t[0]) == len(t[1]))
    if all(_df.comparaison):
        print(f"Les tailles des colonnes {col1} et {col2} sont identiques : {all(_df.comparaison)}")

for col in df.columns:
    if not col in ('Jerk_rot', 'Jerk_pos', 'transition'):
        comparaison(df, "Left_foot", col)

for col in df.columns:
    if not col in ('Jerk_rot', 'Jerk_pos', 'transition'):
        comparaison(df, "Neck_roll", col)

# Analyse de Reduced_state par histogramme de frequence et nuage de points
def printReducedState(df):
    for idx, row in df.iterrows():
        p, v, s, e = row.personne, row.v, row.s, row.e

        Y = row.Reduced_state
        X = np.arange(len(Y))

        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        ax.scatter(X, Y, c=Y, cmap = plt.cm.Set1, marker = 'o')
        plt.savefig(f'./reduced_state/{p}_{s}_{v}_{e}.pdf', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        ax.hist(Y)
        plt.savefig(f'./hist/{p}_{s}_{v}_{e}_hist.pdf', bbox_inches='tight')
        plt.close(fig)
        
printReducedState(df)

# Test de correlation entre Jerk_pos et le temps passe dans la voie
# Ce temps est mesure a partir du la longueur du vecteur X
vitesses = df.personne.to_frame()
vitesses["X"] = df.X.apply(lambda t: t.shape[0])
print(np.corrcoef(df.Jerk_pos,vitesses.X))

plt.plot(vitesses.index, vitesses.X)
plt.show()

# Analyse de hip_roll
def Hip_roll_nuages(df):
    for idx, row in df.iterrows():
        p, v, s, e = row.personne, row.v, row.s, row.e

        Y = row.Hip_roll
        X = np.arange(len(Y))

        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        ax.scatter(X, Y, marker = '.', c = 'navy')
        ax.axhline(y = np.mean(Y), c = 'orangered')
        plt.savefig(f'./Hip_roll/{p}_{s}_{v}_{e}.pdf', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        ax.boxplot(Y)
        plt.savefig(f'./Box_hiproll/{p}_{s}_{v}_{e}_Box_hiproll.pdf', bbox_inches='tight')
        plt.close(fig)

Hip_roll_nuages(df)

"""
#Calcul dérivée Hip Roll
for idx, row in df.iterrows():
        p, v, s, e = row.personne, row.v, row.s, row.e
        
        taille = row.Hip_roll.shape[0]
        derive = np.zeros(taille-1)
        Y = (row.Hip_roll)
        for i in range(taille-1):
          derive[i] = (Y[i+1]-Y[i])/0.01
        print(p,np.mean(abs(derive)))
"""
