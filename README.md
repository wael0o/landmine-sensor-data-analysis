#  Analyse de signaux de mines : Clustering vs Classification supervisée

##  Contexte

Ce projet explore l'analyse de données issues de signaux géophysiques (V, H, S) afin d’identifier différents types de mines.

L’objectif est double :

- Étudier si les données présentent une structure naturellement clusterisable (non supervisé)
- Évaluer la séparabilité des classes via un modèle supervisé

---

##  Données

- Variables explicatives : V, H, S
- Variable cible : M (5 classes)
- Données normalisées
- Format : Excel

---

##  Méthodologie

### 1️ Apprentissage non supervisé

Deux méthodes ont été testées :

- **KMeans**
- **Clustering hiérarchique (Agglomerative, Ward linkage)**

Métrique utilisée :

- **ARI (Adjusted Rand Index)**

Résultats :

| Méthode | ARI |
|----------|------|
| KMeans | ≈ 0.04 |
| Agglomerative | ≈ 0.01 |

-> Les classes ne présentent pas de structure naturellement clusterisable.

---

### 2 Réseau de neurones dense (supervisé)

Architecture :

- Dense 64 (ReLU)
- Dense 32 (ReLU)
- Dense 16 (ReLU)
- Sortie Softmax (5 classes)

Optimisation :

- Adam
- Cross-entropy
- Accuracy

Résultat :

**Accuracy test ≈ 65%**

Hasard = 20%

-> Les données sont partiellement séparables via une frontière décisionnelle non linéaire.

---

##  Analyse des résultats

La matrice de confusion montre :

- Les classes 0 et 1 sont bien séparées
- Les classes 2, 3 et 4 présentent un chevauchement important

Cela explique :

- Le faible ARI en clustering
- La performance modérée du réseau dense

Conclusion :

> Les données ne forment pas de groupes naturels nets,  
> mais une séparation supervisée est possible.

---

##  Compétences mobilisées

- Clustering (KMeans, Ward)
- Réseau de neurones dense (TensorFlow / Keras)
- Analyse de métriques adaptées (ARI vs Accuracy)
- Interprétation de matrice de confusion
- Analyse critique des modèles

---

##  Structure du projet
landmine-project/
    │
    ├── main.py
    │
    ├── data/
    │   └── Mine_Dataset.xls
    │
    └──src/
        ├── data_loader.py
        ├── clustering.py
        ├── deep_model.py
        └── visualization.py



---

##  Conclusion personnelle

Ce projet met en évidence l’importance du choix du modèle et des métriques en fonction de la nature du problème (clustering vs classification).

Il montre qu’un score élevé n’est pas toujours attendu : comprendre la structure des données est essentiel.
