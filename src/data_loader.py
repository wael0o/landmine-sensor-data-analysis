from json import encoder
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#je charge le dataset normalisé
df = pd.read_excel('Mine_Dataset.xls', sheet_name='Normalized_Data',engine='xlrd',decimal=',')

#je separe les variables explicatives de la variable cible
X = df[["V","H","S"]]
Y = df["M"]

#Je choisis n = 5 clusters, car je sais qu'il y a 5 types de mines
kmeans = KMeans(n_clusters=5, n_init=50, random_state=42)
clusters = kmeans.fit_predict(X)

#J'ajoute les clusters au dataframe pour voir la répartition des mines dans les clusters
df['cluster'] = clusters
print(df['cluster'].value_counts())

#Je visualise les clusters
sns.scatterplot(x=df['V'], y=df['H'], hue=df['cluster'], palette='tab10')
plt.show()
sns.scatterplot(x=df['V'], y=df['S'], hue=df['cluster'], palette='tab10')
plt.show()
sns.scatterplot(x=df['H'], y=df['S'], hue=df['cluster'], palette='tab10')
plt.show()

#Je compare les clusters avec la variable cible pour voir si les clusters correspondent aux types de mines
print(pd.crosstab(df['M'], df['cluster']))

#Evaluation de la qualité des clusters avec l'Adjusted Rand Index
ari = adjusted_rand_score(Y, clusters)
print("Adjusted Rand Index:", ari)  

#Conclusion : La methode de KMeans montre des limites dans la classification
#Je test le PCA juste par curiosité

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=Y,
    palette="tab10"
)

plt.title("PCA projection des données")
plt.show()

#Resultats : Le PCA n'apporte pas de separation claire non plus
# 3 tests supplémentaires :
# 1. Test de la méthode DBSCAN
# 2. Test de la méthode Agglomerative Clustering
# 3. Test de la méthode GMM (Gaussian Mixture Models)

#1. Test de la méthode DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.07, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X)

df['cluster_dbscan'] = clusters_dbscan
print(df['cluster_dbscan'].value_counts())
print("Ari DBSCAN:", adjusted_rand_score(Y, clusters_dbscan))

#Visualisation des clusters DBSCAN
sns.scatterplot(x=df["V"], y=df["H"], hue=df["cluster_dbscan"], palette="tab10")
plt.title("DBSCAN V vs H")
plt.show()

sns.scatterplot(x=df["V"], y=df["S"], hue=df["cluster_dbscan"], palette="tab10")
plt.title("DBSCAN V vs S")
plt.show()

sns.scatterplot(x=df["H"], y=df["S"], hue=df["cluster_dbscan"], palette="tab10")
plt.title("DBSCAN H vs S")
plt.show()

#Conclusion : avec un eps de .3 et .07 le ari est de 0.0 
#il ne trouve aucune densité
#pas adapté pour ce dataset

#2. Test de la méthode Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=5,linkage='ward')
clusters_agglo = agglo.fit_predict(X)
df['cluster_agglo'] = clusters_agglo

print(df['cluster_agglo'].value_counts())
print("Ari Agglo:", adjusted_rand_score(Y, clusters_agglo))

#Visualisation des clusters Agglo
sns.scatterplot(x=df["V"], y=df["H"], hue=df["cluster_agglo"], palette="tab10")
plt.title("Agglo V vs H")
plt.show()
sns.scatterplot(x=df["V"], y=df["S"], hue=df["cluster_agglo"], palette="tab10")
plt.title("Agglo V vs S")
plt.show()
sns.scatterplot(x=df["H"], y=df["S"], hue=df["cluster_agglo"], palette="tab10")
plt.title("Agglo H vs S")
plt.show()

#Conclusion : Le ari est de 0.014 legere amelioration
#l'algo segmente plus selon les variables discretes H et S que selon la variable continue V
#Comme il minimise variance intra , il est tres sensible au variables discretes
#Point positif : structure des cluster legeremenet plus claire

#3. Test de la méthode GMM (Gaussian Mixture Models)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)

clusters_gmm = gmm.fit_predict(X)
df['cluster_gmm'] = clusters_gmm

ari_gmm = adjusted_rand_score(Y, clusters_gmm)
print("Ari GMM:", ari_gmm)

#Visualisation des clusters GMM
print(pd.crosstab(df["cluster_gmm"], Y))

sns.scatterplot(x=df["V"], y=df["H"], hue=df["cluster_gmm"], palette="tab10")
plt.title("GMM V vs H")
plt.show()

sns.scatterplot(x=df["V"], y=df["S"], hue=df["cluster_gmm"], palette="tab10")
plt.title("GMM V vs S")
plt.show()

sns.scatterplot(x=df["H"], y=df["S"], hue=df["cluster_gmm"], palette="tab10")
plt.title("GMM H vs S")
plt.show()

#Conclusion : ari = 0.0117, amelioration notable
#GMM suppose des gaussiennes continues or S ne l'est pas voila pourquoi la ari esr si bas selon moi
#Néanmoins GMM segmente bien mieux qu les autres mais bon j'aimerais augmenter le ari

#4. Test methode Mahalanobis + GMM
from sklearn.covariance import EmpiricalCovariance

#on applique la transformation de W pour prendre en compte les correlations entre les variables
cov = EmpiricalCovariance().fit(X)
cov.fit(X)
# Estimation de la covariance
cov = EmpiricalCovariance().fit(X)

# Moyenne
mu = cov.location_

# Matrice de covariance
Sigma = cov.covariance_

# Décomposition spectrale
eigvals, eigvecs = np.linalg.eigh(Sigma)

# Inverse de la racine de Sigma
Sigma_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

# Whitening Mahalanobis
X_maha = (X - mu) @ Sigma_inv_sqrt

gmm_maha = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
clusters_gmm_maha = gmm_maha.fit_predict(X_maha)
df['cluster_gmm_maha'] = clusters_gmm_maha

ari_gmm_maha = adjusted_rand_score(Y, clusters_gmm_maha)
print("Ari GMM Mahalanobis:", ari_gmm_maha) 

#Visualisation des clusters GMM Mahalanobis
sns.scatterplot(x=df["V"],y=df["H"],hue=df["cluster_gmm_maha"],palette="tab10")
plt.title("Mahalanobis + GMM — V vs H")
plt.show()
sns.scatterplot(x=df["V"],y=df["S"],hue=df["cluster_gmm_maha"],palette="tab10")
plt.title("Mahalanobis + GMM — V vs S")
plt.show()
sns.scatterplot(x=df["H"],y=df["S"],hue=df["cluster_gmm_maha"],palette="tab10")
plt.title("Mahalanobis + GMM — H vs S")
plt.show()


#Conclusion : Mauvaise idee, le ari est de 0.0108, diminution du ari

#5. Test methode Autoencoder + GMM
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

input_dim = 3
latent_dim = 2

input_layer = Input(shape=(input_dim,))
h = Dense(64, activation='relu')(input_layer)
z = Dense(latent_dim)(h)

decoder_h = Dense(64, activation='relu')(z)
output_layer = Dense(input_dim)(decoder_h)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=z)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.fit(X,X,epochs = 200, batch_size = 32, verbose=0)

X_latent = encoder.predict(X)

sns.scatterplot(x=X_latent[:,0],y=X_latent[:,1],hue=Y,palette="tab10")
plt.title("Autoencoder projection des données")
plt.show()

gmm_autoencoder = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
clusters_gmm_autoencoder = gmm_autoencoder.fit_predict(X_latent)
df['cluster_gmm_autoencoder'] = clusters_gmm_autoencoder
ari_gmm_autoencoder = adjusted_rand_score(Y, clusters_gmm_autoencoder)
print("Ari GMM Autoencoder:", ari_gmm_autoencoder)
#Visualisation des clusters GMM Autoencoder
sns.scatterplot(x=X_latent[:,0],y=X_latent[:,1],hue=df['cluster_gmm_autoencoder'],palette="tab10")
plt.title("GMM Autoencoder projection des données")
plt.show()
    

#Conclusion : Visuel tres parlant legmentaion est pas mal
#mais mauvais ari, il est negatif

#Conclusion générale :
#Le dataset n'est pas adapté pour du non supervisé

## Partie 2 : Supervisé

#Preparation des données pour du supervisé
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=42,stratify=Y)

#1 Logistic Regression
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(multi_class='multinomial', max_iter=1000)
logreg.fit(X_train_scaled, Y_train)
Y_pred_logreg = logreg.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred_logreg))
print(confusion_matrix(Y_test, Y_pred_logreg))
print(classification_report(Y_test, Y_pred_logreg))

#Conclusion : pour du supervisé on parle d'accuracy et pas de ari
# accuracy = 0.47
#Les minies de  classe 2 et 1  facilement identifiable, cependant pour les autre classes
#le modele a un peu plus de mal, fort chevauchement entre les classes 0,3 et 4


#je test d'autres modeles pour voir si je peux faire mieux que 0.47

#2. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300,max_depth=None, random_state=42)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))
print(confusion_matrix(Y_test, Y_pred_rf))
print(classification_report(Y_test, Y_pred_rf))

#Conclusion : accuracy de 0.47, auxune amleioration, ça plafonne 

#3. Support Vector Machine (SVM)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, Y_train)
Y_pred_svm = svm.predict(X_test_scaled)

print("SVM Accuracy:", accuracy_score(Y_test, Y_pred_svm))
print(confusion_matrix(Y_test, Y_pred_svm))
print(classification_report(Y_test, Y_pred_svm))

#COnclsion : on monte 0.51, c'est mieu. data insiffisante



#ajout d'un graph en 3d

from mpl_toolkits.mplot3d import Axes3D

def plot_3d(df, x="V", y="H", z="S", hue="M", title="3D scatter"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    values = df[hue]
    labels, uniques = pd.factorize(values)
    cmap = plt.get_cmap("tab10")
    colors = cmap(labels % 10)

    ax.scatter(df[x], df[y], df[z], c=colors, s=40, alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=str(u),
                   markerfacecolor=cmap(i % 10), markersize=8)
        for i, u in enumerate(uniques)
    ]
    ax.legend(handles=handles, title=hue)
    plt.show()

# appel
plot_3d(df, x="V", y="H", z="S", hue="M", title="Mines 3D (V, H, S)")
