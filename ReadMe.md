# Projet FocusSAR

## Liste de fichiers et de leur utilité

```ProcessingSAR.py```

Fichier permettant d'effectuer le prétraitement des données brutes SAR.
Le PATH vers le fichier contenant les données brutes est à indiquer ligne 28.
Si le fichier est modifié, il est possible que des modifications soient à 
effectuer au niveau des paramètres.

```SaveSARdtat.py```

Fichier permettant de lire des données SAR prétraitées au format NITF et 
de les enregistrer au format MAT.

```Util.py```

Fichier contenant des fonctions utiles à la réalisation de l'Autofocus et 
communes aux différentes méthodes implémentées.

```MEA.py```

Fichier contenant la chaine de traitement permettant de réaliser le Minimum 
Entropy Autofocus comme décrit dans <em>"SAR Minimum-Entropy Autofocus Using 
an Adaptive-Order Polynomial Model"</em>.

Les lignes 202 et 203 permettent de choisir si on réalise l'Autofocus sur 
les données réelles ou simulées.

La ligne 206 permet de choisir le type d'erreur de phase à appliquer sur 
l'image SAR.

La ligne 207 permet de choisir le niveau de bruit (en décibel) de l'image
défocus.

```PEA.py```

Fichier contenant la chaine de traitement permettant de réaliser le Phase 
Error Autofocus comme décrit dans <em>"Feature Preserving Autofocus 
Algorithm for Phase Error Correction of SAR Images"</em>.

Les lignes 79 et 80 permettent de choisir si on réalise l'Autofocus sur 
les données réelles ou simulées.

La ligne 85 permet de choisir le type d'erreur de phase à appliquer sur 
l'image SAR.

La ligne 86 permet de choisir le niveau de bruit (en décibel) de l'image
défocus.

```PGA.py```

Fichier contenant la chaine de traitement permettant de réaliser le Phase 
Gradient Autofocus comme décrit dans <em>"Phase Gradient Autofocusing 
Technique (PGA)"</em>.

La méthode n'est pas fonctionnelle telle qu'elle est implémentée.

Les lignes 194 et 195 permettent de choisir si on réalise l'Autofocus sur 
les données réelles ou simulées.

La ligne 198 permet de choisir le type d'erreur de phase à appliquer sur 
l'image SAR.

La ligne 199 permet de choisir le niveau de bruit (en décibel) de l'image
défocus.

## Liste des librairies Python nécessaires et version de python utilisée pendant le projet

Les versions 3.10 et 3.11 de Python a été utilisée pendant ce projet.

Les librairies suivantes sont utilisées das les différents fichiers cités 
précédemment :

```time```

```numpy```

```scipy```

```matplotlib```

```sarpy```

## Source des données réelles utilisées

https://github.com/ngageoint/six-library/wiki/Sample-SICDs

