# Fake Job Postings Detection – NLP Project

Ce projet a pour objectif de détecter les offres d’emploi frauduleuses à partir du texte
en utilisant des techniques de traitement automatique du langage naturel (NLP)
et des modèles de machine learning.

---

## Partitionnement du projet

Le projet a été réparti en quatre parties distinctes afin de faciliter l’organisation
du travail et de clarifier les responsabilités de chaque membre de l’équipe.

- **Partie 1 : Data Cleaning (Étapes 1 à 3)**  
  Cette partie comprend le chargement des données, le nettoyage du texte
  et la suppression des doublons.  
  Elle a été réalisée par **HALLA Hajar**.

- **Partie 2 : Vectorisation et Modélisation SVM (Étapes 4 à 10)**  
  Cette partie couvre la construction du texte, la sélection des variables,
  l’échantillonnage, la séparation apprentissage/test, la vectorisation TF-IDF,
  la réduction de dimension par SVD (PCA) et l’entraînement du modèle SVM linéaire.  
  Elle a été réalisée par **ELAZAOUI Maroua**.

- **Partie 3 : Modèle KNN (Étapes 10 à 11 + étape 13)**  
  Cette partie consiste à appliquer le modèle KNN sur les données vectorisées,
  à tester différentes valeurs de K et à sélectionner le meilleur modèle
  selon le F1-score.  
  Elle a été réalisée par **Lahrach Nouhaila**.

- **Partie 4 : Embeddings pré-entraînés et évaluation finale (Étapes 12 à 15)**  
  Cette partie inclut l’utilisation des embeddings SBERT, la classification
  avec KNN basé sur la distance cosinus, l’évaluation des performances
  et l’export des résultats finaux.  
  Elle a été réalisée par **ELMOUBACHOUR Oumiama**.

---

## Étape 1 — Chargement des données
Le dataset est chargé depuis un fichier ZIP contenant un fichier CSV.
Le nombre initial d’observations est enregistré pour suivi.

---

## Étape 2 — Nettoyage des données
Les balises HTML sont supprimées du texte.
Les espaces inutiles sont normalisés.
Les valeurs manquantes dans les champs textuels sont remplacées par des chaînes vides.

---

## Étape 3 — Suppression des doublons
Les doublons sont supprimés en se basant sur l’identifiant de l’offre (`job_id`).
Les lignes entièrement dupliquées sont également retirées.

---

## Étape 4 — Construction du texte
Les colonnes textuelles importantes (titre, description, entreprise, exigences, avantages)
sont concaténées pour former un document texte unique par offre.
Les offres avec un texte trop court sont supprimées.

---

## Étape 5 — Sélection des variables utiles
Seules les colonnes pertinentes pour l’analyse sont conservées.
La variable cible est `fraudulent` (0 : non frauduleuse, 1 : frauduleuse).

---

## Étape 6 — Échantillonnage
Un échantillon de 10 000 offres est sélectionné de manière aléatoire
afin de réduire le temps de calcul.

---

## Étape 7 — Séparation apprentissage / test
Les données sont séparées en un ensemble d’apprentissage (80 %)
et un ensemble de test (20 %) de manière stratifiée.

---

## Étape 8 — Vectorisation TF-IDF
Les textes sont transformés en vecteurs numériques à l’aide de TF-IDF.
Les unigrammes et bigrammes sont pris en compte.
Les mots trop rares ou trop fréquents sont ignorés.

---

## Étape 9 — Réduction de dimension (SVD / PCA)
Une réduction de dimension est appliquée à l’aide de TruncatedSVD.
Cette étape permet de conserver l’essentiel de l’information
tout en réduisant la complexité du modèle.

---

## Étape 10 — Modèle SVM linéaire
Un SVM linéaire est entraîné sur les données réduites.
Une validation croisée est utilisée pour choisir le meilleur paramètre C.
La métrique principale utilisée est le F1-score.

---

## Étape 11 — Modèle KNN (comparaison)
Un modèle KNN est entraîné sur les mêmes données.
Plusieurs valeurs de K sont testées.
Le meilleur K est sélectionné selon le F1-score sur l’ensemble de test.

---

## Étape 12 — Embeddings pré-entraînés (SBERT)
Un modèle SBERT pré-entraîné est utilisé pour représenter chaque document.
Chaque texte est découpé en segments.
L’embedding final du document est la moyenne des embeddings des segments.

---

## Étape 13 — Classification avec SBERT + KNN
Un modèle KNN avec la distance cosinus est appliqué sur les embeddings SBERT.
Cette approche permet de comparer les performances avec TF-IDF.

---

## Étape 14 — Évaluation des modèles
Les performances sont évaluées à l’aide de :
- Accuracy
- F1-score
- Classification report

---

## Étape 15 — Export des résultats
Les prédictions finales du modèle SVM sont exportées dans un fichier CSV.
Le fichier contient le texte de l’offre, le vrai label et la prédiction du modèle.

---

## Conclusion
Ce projet montre l’intérêt des méthodes NLP classiques (TF-IDF)
et des modèles pré-entraînés (SBERT) pour la détection d’offres d’emploi frauduleuses.
Les résultats permettent de comparer différentes approches
en termes de performance et de coût de calcul.

---

## Auteur
Projet académique – INSEA
.
