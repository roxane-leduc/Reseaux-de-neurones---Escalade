# Reseaux-de-neurones---Escalade
Mise en œuvre des techniques d’apprentissage statistique appliquées à des données provenant de sportifs faisant de l’escalade dans des conditions variées.

Ce projet a été réalisé en Python et comprend plusieurs modules qui nécessitent l'installation de différentes bibliothèques. 
Le projet est axé sur l'analyse descriptive des données et l'utilisation de réseaux de neurones. 
Voici une description des différents modules :

1) import_jeu_donnes.py
Ce module permet de créer un dataframe en chargeant les données et en sélectionnant uniquement les informations pertinentes pour notre analyse.

Remarque: Les données proviennent du dossier data.

2) stat_descriptive.py
Ce module comprend une partie de l'analyse descriptive des données. Il propose plusieurs fonctionnalités telles que la création de boîtes à moustaches, le tracé de Y en fonction de X, le calcul de la corrélation entre les jerk, l'affichage d'histogrammes de fréquence et la création de nuages de points.

Remarque: Pour exécuter correctement ce module, il est nécessaire de créer les dossiers suivants : Box_hiproll, Hip_roll, img, hist et reduced_state. Ces dossiers servent à stocker les résultats générés par les différentes analyses effectuées dans le module.

3) kmeans_rdn_vf.py
Ce module comprend la suite de l'analyse descriptive des données, y compris la génération d'un dendrogramme, l'utilisation de la méthode Elbow, l'application de l'algorithme K-means et l'analyse par classe. Ce module inclut également une section dédiée aux réseaux de neurones.

Remarque : Certains passages sont commentés afin d'optimiser les temps de compilation. L'utilisateur peut choisir de décommenter les sections qu'il souhaite exécuter en fonction de ses besoins.
