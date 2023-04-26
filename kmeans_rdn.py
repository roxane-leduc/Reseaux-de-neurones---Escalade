import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pandas import DataFrame
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram, linkage

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

# ====================================================================
# IMPORT DU JEU DE DONNEES
# ====================================================================

df = load_datasets()
dfOrig = df.copy(deep=True)
df = clean_datasets(df)

# ====================================================================
# ANALYSE DU JEU DE DONNEES
# ====================================================================

# Construction d'un DENDROGRAMME (base sur la vitesse)
df["vitesse"] = df.X.apply(lambda v: v.shape[0])
df["vitesse"] = df.Reduced_state.apply(lambda v: v.shape[0])
# df["vitesse"] = df.Jerk_rot
# df["vitesse"] = df.Jerk_pos

Z = linkage(df.vitesse.to_frame(), method='ward', metric='euclidean')
d = dendrogram(Z, orientation='top', labels=df.index, show_contracted=True, 
	no_plot=False, leaf_font_size=6, leaf_rotation=45.)
plt.show()


# Compter le nb de grimpeur en regroupant par s et v:
# print(df.groupby(by=["s", "v"]).personne.count())

# ANOMALIE DE MESURE : df.loc[58]
# Reduced_state, Left_foot... SONT DES SINGLETONS
df = df[df.index != 58].copy(deep=True)
df.reset_index(drop=True, inplace=True)

df["vitesse"] = df.Reduced_state.apply(lambda v: v.shape[0])

# r = df[["s", "v", "vitesse"]].groupby(by=["s", "v"]).agg({"vitesse" : ["count", np.min, np.mean, np.max, np.std]})
# print(r)

# ELBOW METHOD (nb optimal de cluster)
from sklearn.cluster import KMeans

_df = df.copy(deep=True)
for i in range(5):
    _df[f"Reduced_state{i}"] = df.Reduced_state.apply(lambda v: len([x for x in v if x == i]))
# _df = _df[["vitesse","Jerk_rot", "Reduced_state0", "Reduced_state1", "Reduced_state2", "Reduced_state3", "Reduced_state4"]].copy(deep=True)
_df = _df[["vitesse", "Reduced_state0", "Reduced_state1", "Reduced_state2", "Reduced_state3", "Reduced_state4"]].copy(deep=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
_df = scaler.fit_transform(_df)

Sum_of_squared_distances, K = [], range(1, 10)
for nclusters in K:
	kmeans = KMeans(n_clusters=nclusters)
	kmeans.fit(_df)
	Sum_of_squared_distances.append(kmeans.inertia_)
	
plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("Nb of clusters") 
plt.ylabel("Sum of squared distances / Inertia") 
plt.title("Elbow Method For Optimal k")
plt.show()

# KMEANS (permet de creer les labels y pour le rdn)
n_clusters=3
kmeans = KMeans(n_clusters, n_init=50, random_state=0)
kmeans.fit(_df)
df["CategorieGrimpeur"] = kmeans.labels_
print(len(kmeans.labels_))

print(df.groupby(by="CategorieGrimpeur").vitesse.count())

# ====================================================================
# RESEAU DE NEURONES
# ====================================================================

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

nTimeStep = df.Reduced_state.apply(lambda v: len(v)).max()

df2 = df[["Reduced_state", "Left_foot", "Hip_roll", "CategorieGrimpeur"]].copy(deep=True)
df2.rename(columns={"Reduced_state": "X1", "Left_foot": "X2", "Hip_roll": "X3", "CategorieGrimpeur": "y"}, inplace=True)

# (truncate and) pad input sequences
df2.X1 = df2.X1.apply(lambda t: np.pad(t, (0, nTimeStep - len(t))))
df2.X2 = df2.X2.apply(lambda t: np.pad(t, (0, nTimeStep - len(t))))
df2.X3 = df2.X3.apply(lambda t: np.pad(t, (0, nTimeStep - len(t))))
df2["X"] = df2.apply(lambda row: np.asarray([row.X1, row.X2, row.X3]), axis=1)
X = df2.apply(lambda row: np.asarray([row.X1, row.X2, row.X3]).T, axis=1).to_list()

#X=df2[['X1', 'X2', 'X3']].values
y = df2['y'].values

# Convertir les labels en codage one-hot
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Convertir les tableaux NumPy en tenseurs TensorFlow
X_train = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(X_val)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)

# Définir le modèle LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(6, input_shape=(nTimeStep, 3)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compiler le modèle avec une fonction de perte, un optimiseur et une métrique d'évaluation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle sur les données d'entraînement
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))

# Évaluer le modèle sur les données de test
loss, accuracy = model.evaluate(X_val, y_val)

# Faire des prédictions avec le modèle
y_pred = model.predict(X_val)

# Créer un graphique
plt.figure(figsize=(10, 5))
plt.plot(y_val, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Prédictions LSTM')
plt.xlabel('Temps')
plt.ylabel('Valeurs de y')
plt.legend()
plt.style.use('ggplot')
plt.show()