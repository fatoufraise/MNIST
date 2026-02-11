# MNIST

Ce projet a pour objectif de mettre en œuvre un système simple de reconnaissance automatique de chiffres manuscrits à l’aide d’un réseau de neurones convolutifs (CNN).
Le système repose sur la bibliothèque TensorFlow et utilise le jeu de données MNIST, qui constitue une référence standard dans le domaine de l’apprentissage automatique et de la vision par ordinateur.

L’application permet à l’utilisateur de dessiner un chiffre à l’aide d’une interface graphique, puis d’obtenir une prédiction automatique du chiffre saisi.


# Technologies utilisées

Langage : Python 3.10
Framework de deep learning : TensorFlow (tensorflow-macos)
Accélération matérielle : tensorflow-metal (GPU Apple Silicon)
Interface graphique : Tkinter
Bibliothèques supplémentaires :

  - NumPy
  - Pillow
  - Matplotlib


Le projet utilise le jeu de données MNIST, composé de :

- 70 000 images en niveaux de gris
- Taille des images : 28 × 28 pixels
- 10 classes correspondant aux chiffres de 0 à 9

Chaque image est associée à une étiquette représentant le chiffre manuscrit correspondant.



# Installation et exécution (sur le terminal)


conda create -n mnist_tf python=3.10 -y
conda activate mnist_tf


# Installation des dépendances


pip install --upgrade pip
pip install "numpy<1.25" tensorflow-macos==2.13 tensorflow-metal==0.14 pillow matplotlib


# Entraînement du modèle

L’entraînement du modèle est effectué à l’aide du script suivant :


python train_model.py


À l’issue de l’entraînement, le modèle est sauvegardé dans le dossier "models/".



# Utilisation de l’application

L’application graphique est lancée avec la commande suivante :


python main.py


L’utilisateur peut alors :

1. Dessiner un chiffre à la souris
2. Lancer la prédiction
3. Observer le chiffre reconnu par le modèle


# Résultats attendus

Le modèle atteint une précision élevée sur le jeu de test MNIST (> 98 %).
Les prédictions sur les chiffres dessinés par l’utilisateur sont généralement correctes, bien que dépendantes de la qualité du tracé.


# Limites du projet

- Reconnaissance limitée à un seul chiffre à la fois
- Sensibilité à la qualité du dessin
- Interface graphique volontairement simple


# Perspectives d’amélioration

Plusieurs améliorations peuvent être envisagées :

- Reconnaissance de séquences de chiffres
- Amélioration de l’interface utilisateur
- Entraînement sur des données personnalisées
- Déploiement sous forme d’application autonome


