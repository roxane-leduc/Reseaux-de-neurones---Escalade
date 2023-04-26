import matplotlib.pyplot as plt
from glob import glob
from pandas import DataFrame
from scipy.io import loadmat

# ====================================================================
def load_datasets():
    rows = []
    for filename in glob('data/G*.mat'):
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
df = load_datasets()
dfOrig = df.copy(deep=True)

# NETTOYAGE DU JEU DE DONNEES
df = clean_datasets(df)

# print(df.columns.values)
# df.Hip_roll.apply(lambda v: v.max()).max()
#df.Hip_roll.apply(lambda v: v.min()).min()

"""
for nRow, row in df.iterrows():
    sizeOfTheVectorValue = row.Reduced_state.size
    for fieldname in ['Reduced_state', 'Neck_local_var', 'Hip_local_var', 'Hip_roll', 'Right_hand', 'Neck_roll', 'Left_foot', 'Y', 'X', 'Roll_correl', 'Left_hand', 'Afford_count', 'Right_foot']:
        # Pour une ligne donnee les vecteurs des colonnes de la boucle ont tous la même taille ?
        if (sizeOfTheVectorValue != row[fieldname].size):
            print(f"Inconsistance en ligne {nRow} avec la colonne {fieldname}")
"""

