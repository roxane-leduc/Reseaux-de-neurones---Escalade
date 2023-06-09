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
# ANALYSE DESCRIPTIVE DU JEU DE DONNEES 2
# ====================================================================

# DENDROGRAMME (base sur la vitesse)
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
print(type(_df),type(df), "????????????????????????????????????????????????????????????????")
n_clusters=3
kmeans = KMeans(n_clusters, n_init=50, random_state=0)
kmeans.fit(_df)
df["CategorieGrimpeur"] = kmeans.labels_
#print(len(kmeans.labels_))


# ====================================================================
# ANALYSE DESCRIPTIVE DU JEU DE DONNEES 3
# ====================================================================

# Analyse par classe
df_0 = df[df.CategorieGrimpeur == 0]
df_1 = df[df.CategorieGrimpeur == 1]
df_2 = df[df.CategorieGrimpeur == 2]

df_0 = df_0.reset_index(drop = True)
df_1 = df_1.reset_index(drop = True)
df_2 = df_2.reset_index(drop = True)

n0 = len(df_0)
n1 = len(df_1)
n2 = len(df_2)

#VITESSE
v = 0
for j in range(n0):
    v +=  df_0.vitesse[j]
v = v/n0
print("Temps de grimpe moyen du premier groupe : ", v)

v=0
for j in range(n1):
    v += df_1.vitesse[j]
v = v/n1
print("Temps de grimpe moyen du deuxième groupe : ", v)

v = 0
for j in range(n2):
    v += df_2.vitesse[j]
v = v/n2
print("Temps de grimpe moyen du troisième groupe : ", v)

"""
#Jerk

v = 0
for j in range(n0):
    v +=  df_0.Jerk_rot[j]
v = v/n0
print("Jerk moyen du premier groupe : ", f"{v:.2e}")

v=0
for j in range(n1):
    v += df_1.Jerk_rot[j]
v = v/n1
print("Jerk moyen du deuxième groupe : ", f"{v:.2e}")

v = 0
for j in range(n2):
    v += df_2.Jerk_rot[j]
v = v/n2
print("Jerk moyen du troisième groupe : ", f"{v:.2e}")

#REDUCED STATE

print(df_0.Reduced_state[0].shape, "................")


# Premier groupe
c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
for i in range(n0):
    for j in range(df_0.Reduced_state[i].shape[0]):
        if (df_0.Reduced_state[i][j] == 0):
            c0 += 1
        if (df_0.Reduced_state[i][j] == 1):
            c1 += 1
        if (df_0.Reduced_state[i][j] == 2):
            c2 += 1
        if (df_0.Reduced_state[i][j] == 3):
            c3 += 1
        if (df_0.Reduced_state[i][j] == 4):
            c4 += 1

X_0 = [c0, c1, c2, c3, c4]

# Deuxième groupe
c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
for i in range(n1):
    for j in range(df_1.Reduced_state[i].shape[0]):
        if (df_1.Reduced_state[i][j] == 0):
            c0 += 1
        if (df_1.Reduced_state[i][j] == 1):
            c1 += 1
        if (df_1.Reduced_state[i][j] == 2):
            c2 += 1
        if (df_1.Reduced_state[i][j] == 3):
            c3 += 1
        if (df_1.Reduced_state[i][j] == 4):
            c4 += 1


X_1 = [c0, c1, c2, c3, c4]

# Troisième groupe 
c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
for i in range(n2):
    for j in range(df_2.Reduced_state[i].shape[0]):
        if (df_2.Reduced_state[i][j] == 0):
            c0 += 1
        if (df_2.Reduced_state[i][j] == 1):
            c1 += 1
        if (df_2.Reduced_state[i][j] == 2):
            c2 += 1
        if (df_2.Reduced_state[i][j] == 3):
            c3 += 1
        if (df_2.Reduced_state[i][j] == 4):
            c4 += 1

X_2 = [c0, c1, c2, c3, c4]

X = [0, 1, 2 , 3, 4]

fig, ax = plt.subplots(figsize=(8.26, 8.26))
ax.scatter(X,X_0, color = 'salmon', label = 'groupe 1')
ax.scatter(X,X_1, color = 'red', label = 'groupe 2')
ax.scatter(X, X_2, color = 'b', label = 'groupe 3')
ax.legend()
plt.title("Moyenne des états du grimppeur (reduced state) par groupe")
plt.savefig(f'./hist_ReducedState/histo.pdf', bbox_inches='tight')
plt.close(fig)

# HIP ROLL

hp_0 = []
hp_1 = []
hp_2 = []

for i in range(n0):
    for j in range(df_0.Hip_roll[i].shape[0]):
        hp_0.append(df_0.Hip_roll[i][j])

for i in range(n1):
    for j in range(df_1.Hip_roll[i].shape[0]):
        hp_1.append(df_1.Hip_roll[i][j])
        
for i in range(n2):
    for j in range(df_2.Hip_roll[i].shape[0]):
        hp_2.append(df_2.Hip_roll[i][j])

all_hp = [hp_0, hp_1, hp_2]
colors = ['salmon','red','blue']
fig, ax = plt.subplots(figsize=(8.26, 8.26))
bp1 = ax.boxplot(all_hp, vert=True, patch_artist = True, labels = ['Groupe1', 'Groupe2', 'Groupe 3'])

for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    

plt.title("Boites à moustache pour l'orientation des hanches (hip_roll) par groupe")
plt.savefig(f'./Box_hiproll/boxplo.pdf', bbox_inches='tight')
plt.close(fig)

"""

# ====================================================================
# RESEAU DE NEURONES (MODELE LSTM)
# ====================================================================

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

nTimeStep = df.Reduced_state.apply(lambda v: len(v)).max()

# Mettre la vitesse en série temporelle
vitesse_sequence = []
for i in range(104):
    l = []
    for j in range(nTimeStep):
        l.append(df["vitesse"][i])
    vitesse_sequence.append(l)

df["vitesseSeq"] = vitesse_sequence  

# Observer la distribution des variables
def plotDistribGraph(pdf):
    fig, a = plt.subplots(ncols=1, figsize=(16, 5))
    a.set_title("Distributions")
    for col in pdf.columns:
        taille = pdf[col][0].shape[0]
        tab = np.zeros(104*taille)
        for j in range(104):
            for i in range(taille):
                tab[taille + j*104] = pdf[col][j][i]
        sns.kdeplot(tab, ax=a)
    plt.show()

def plotGraph(pdf, pscaled_df):
    fig, (a, b) = plt.subplots(ncols=2, figsize=(16, 5))
    a.set_title("Avant mise à l'echelle")
    for col in pdf.columns:
        sns.kdeplot(pdf, ax=a)
    b.set_title("Apres mise à l'echelle")
    for col in pdf.columns:
        sns.kdeplot(pscaled_df[col], ax=b)
    plt.show()

"""
#Fonction d activation finale
def fct_activ(x):
    return tf.nn.leaky_relu(x, alpha=0.01)
"""

df2 = df[["Reduced_state", "vitesseSeq", "Hip_roll" , "CategorieGrimpeur"]].copy(deep=True)

#t = df2.to_numpy()
#scaler.fit_transform(t)
df2.rename(columns={"Reduced_state": "X1", "vitesseSeq": "X2","Hip_roll": "X3", "CategorieGrimpeur": "y"}, inplace=True)

# (truncate and) pad input sequences
df2.X1 = df2.X1.apply(lambda t: np.pad(t, (0, nTimeStep - len(t)), mode = 'linear_ramp', end_values=(0, 4)))
df2.X2 = df2.X2.apply(lambda t: np.pad(t, (0, nTimeStep - len(t)), mode = 'mean'))
df2.X3 = df2.X3.apply(lambda t: np.pad(t, (0, nTimeStep - len(t)), mode = 'linear_ramp', end_values=(-1.8, 1.8)))
df2.X2 = df2.X2.apply(lambda t: t/10000)

df2["X"] = df2.apply(lambda row: np.asarray([row.X1, row.X2, row.X3]), axis=1)
df2 = df2[["X1","X3","y"]]
X = df2.apply(lambda row: np.asarray([row.X1, row.X3]).T, axis=1).to_list()

#print("VERIF 2 : ", df2['X1'][0])

"""
# Mise a l echelle
df3 = df2[["X1", "X2", "X3" ]].copy(deep=True)

df4 = pd.DataFrame(columns = ['X1','X2','X3'])
for col in df3.columns:
        taille = df3[col][0].shape[0]
        tab = np.zeros(104*taille)
        for j in range(104):
            for i in range(taille):
                tab[taille + j*104] = df3[col][j][i]
        df4[col] = tab
        
print(df4['X1'])
	
scaler = RobustScaler()
scaled_df = scaler.fit_transform(df4)
scaled_df = pd.DataFrame(scaled_df, columns=["X1", "X2", "X3" ])

for col in scaled_df.columns:
    c = 0
    l = []
    for i in range(104):
        l2 = []
        for j in range(taille*i, taille*(i+1)):
            l2.append(scaled_df[col][j])
        l.append(l2)
    df3[col] = l

plotGraph(df4,scaled_df)    
                
#X = df3.apply(lambda row: np.asarray([row.X1, row.X2, row.X3]).T, axis=1).to_list()   
    
"""

"""
#print(df2[0],"?????????")
#X=df2[['X1', 'X2', 'X3']].values
y = df2['y'].values

# Convertir les labels en codage one-hot
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

#print(len(X_train), len(X_val), len(y_train) , len(y_val))

# Convertir les tableaux NumPy en tenseurs TensorFlow
X_train = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(X_val)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)


# Définir le modèle LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, kernel_initializer='random_normal', activation = 'relu', return_sequences = True, input_shape=(nTimeStep, 2)),
    tf.keras.layers.LSTM(10, activation = 'relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
   
])

# Compiler le modèle avec une fonction de perte, un optimiseur et une métrique d'évaluation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Stocker les meilleurs poids
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_accuracy',
                               verbose=1, 
                               save_best_only=True)

# Entraîner le modèle sur les données d'entraînement
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

#mon_fichier = h5py.File('./best_weights.hdf5', 'a')
model.save_weights('best_weights.hdf5')

#model.load_weights('best_weights.hdf5')

# Évaluer le modèle sur les données de test
loss, accuracy = model.evaluate(X_val, y_val)

print("Accuracy :", accuracy)
print("Loss :", loss)
model.summary()

# Faire des prédictions avec le modèle
y_pred = model.predict(X_val)

#print(X_val)
print(y_pred)
print(y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label="Accuracy d'entraînement")
plt.plot(epochs, val_acc, 'royalblue', label='Accuracy de validation')
plt.title("Accuracy d'entraînement et de validation")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label="Perte d'entraînement")
plt.plot(epochs, val_loss, 'royalblue', label='Perte de validation')
plt.title("Perte d'entraînement et de validation")
plt.legend()
plt.show()

"""
# ====================================================================
# RESEAU DE NEURONES DENSE
# ====================================================================

df5 = df[["vitesse", "Jerk_rot", "CategorieGrimpeur"]].copy(deep=True)

#t = df2.to_numpy()
#scaler.fit_transform(t)
df5.rename(columns={"vitesse": "X1", "Jerk_rot": "X2", "CategorieGrimpeur": "y"}, inplace=True)

# (truncate and) pad input sequences

df5.X2 = df5.X2.apply(lambda t: t/100000000000)
df5["X"] = df5.apply(lambda row: np.asarray([row.X1, row.X2]), axis=1)

df5 = df5[["X1","X2","y"]]
X = df5.apply(lambda row: np.asarray([row.X1, row.X2]).T, axis=1).to_list()

y = df5['y'].values

# Convertir les labels en codage one-hot
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

#print(len(X_train), len(X_val), len(y_train) , len(y_val))

# Convertir les tableaux NumPy en tenseurs TensorFlow
X_train = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(X_val)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)

# Définir le modèle LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, kernel_initializer='random_normal', activation = 'relu', input_shape=(None, 2)),
    tf.keras.layers.Dense(30, activation = 'relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])

# Compiler le modèle avec une fonction de perte, un optimiseur et une métrique d'évaluation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle sur les données d'entraînement
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Évaluer le modèle sur les données de test
loss, accuracy = model.evaluate(X_val, y_val)

print("Accuracy :", accuracy)
print("Loss :", loss)
model.summary()

# Faire des prédictions avec le modèle
y_pred = model.predict(X_val)

#print(X_val)
print(y_pred)
print(y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label="Accuracy d'entraînement")
plt.plot(epochs, val_acc, 'royalblue', label='Accuracy de validation')
plt.title("Accuracy d'entraînement et de validation")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label="Perte d'entraînement")
plt.plot(epochs, val_loss, 'royalblue', label='Perte de validation')
plt.title("Perte d'entraînement et de validation")
plt.legend()
plt.show()

"""
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

"""
