# Analyse de données issues de capteurs de détection de mines

Ce projet s’inscrit dans un cadre d’analyse de données appliquée à la détection de mines terrestres. À partir de mesures issues de capteurs (V, H, S), l’objectif est d’évaluer dans quelle mesure ces variables permettent de caractériser et de distinguer différents types de mines. Le travail a été mené dans une logique d’analyse progressive, combinant exploration, modélisation et interprétation des limites du jeu de données.

## Données

Le jeu de données est composé de mesures normalisées :
- **V** : tension mesurée par le capteur  
- **H** : paramètre lié à la configuration ou à la hauteur de mesure  
- **S** : paramètre décrivant le sol  
- **M** : type de mine (variable cible)

Les variables explicatives décrivent des conditions expérimentales et physiques, certaines étant continues, d’autres discrètes, ce qui influence fortement la structure géométrique des données.

## Méthodologie

L’analyse a été menée en deux temps.

Dans un premier temps, des méthodes d’apprentissage non supervisé ont été utilisées afin d’explorer l’existence éventuelle de structures naturelles dans les données, indépendamment des types de mines connus. Plusieurs approches ont été testées (K-Means, DBSCAN, clustering hiérarchique, modèles de mélanges gaussiens), ainsi que des méthodes de réduction de dimension (ACP, autoencoder). Les résultats ont été évalués à l’aide de l’Adjusted Rand Index (ARI) et interprétés à l’aide de visualisations en deux et trois dimensions.

Dans un second temps, des modèles supervisés ont été entraînés afin d’estimer la capacité maximale de discrimination permise par les variables disponibles. Des modèles linéaires et non linéaires ont été comparés (régression logistique, Random Forest, SVM), avec une évaluation basée sur l’accuracy, les matrices de confusion et les rapports de classification.

## Résultats

Les méthodes non supervisées ne mettent pas en évidence de clusters stables correspondant aux cinq types de mines. Les scores ARI restent faibles quelle que soit la méthode employée, ce qui suggère un fort chevauchement des signatures dans l’espace des variables. Les visualisations 2D et 3D confirment cette observation, en montrant des zones d’ambiguïté persistantes.

Les modèles supervisés permettent de dépasser largement le hasard, avec des performances atteignant environ 50 % d’accuracy pour les meilleurs modèles. Certaines classes présentent des signatures relativement stables et sont bien identifiées, tandis que d’autres restent fortement confondues, y compris avec des modèles non linéaires puissants.

La cohérence entre les résultats supervisés, non supervisés et visuels indique que ces limites ne sont pas liées au choix des algorithmes, mais à l’information contenue dans les données elles-mêmes.

## Approche analytique

Plutôt que de chercher à tout prix à améliorer un score ou à forcer une classification parfaite, l’analyse a consisté à évaluer ce que les données permettent réellement de conclure. L’absence de séparation claire en non supervisé, combinée au plafonnement des performances en supervisé, a été interprétée comme le signe d’une limite informationnelle.

Cette démarche vise à distinguer un échec algorithmique d’une impossibilité structurelle. Les algorithmes ont ici servi d’outils de diagnostic : ils montrent que certaines signatures sont intrinsèquement ambiguës dans l’espace (V, H, S), tandis que d’autres sont plus stables et exploitables.

## Conclusion analytique

Ce travail montre que les variables V, H et S contiennent une information partielle mais insuffisante pour reconstruire de manière fiable la typologie complète des mines définie par la variable M. La difficulté rencontrée n’est pas liée à un manque de sophistication des modèles, mais à un chevauchement structurel des signatures physiques mesurées.

D’un point de vue analytique et opérationnel, ces résultats suggèrent que les données étudiées sont davantage adaptées à une logique d’aide à la décision — détection, priorisation, estimation de l’incertitude — qu’à une classification exhaustive et déterministe. Cette conclusion met en évidence l’importance de reformuler les objectifs d’analyse en fonction de l’information réellement accessible, plutôt que de supposer a priori que toute typologie est reconstructible à partir des données disponibles.

## Outils utilisés

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, TensorFlow / Keras.
