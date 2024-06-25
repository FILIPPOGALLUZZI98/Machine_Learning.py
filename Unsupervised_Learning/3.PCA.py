import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Supponiamo di avere il dataset X contenente dei punti di tipo (x1, x2, x3)

# Facciamo la scomposizione in PCA
# In questo caso mettiamo le PCs uguali al numero di coordinate iniziali (in questo caso = 3)
# per vedere le diverse explained variances
pca_3 = PCA(n_components=3)
pca_3.fit(X)  ## In questo caso non dobbiamo fare la normalizzazione dato che sklearn già lo fa

# Vediamo ora la explained variance per gli assi 
pca_3.explained_variance_ratio_  ## Vettore con 3 coordinate che sono le explained variance ratio

# Se una coordinata ha explained variance molto più grande delle altre, riduciamo le coordinate
# Supponiamo che sia (0.9, 0.05, 0.05). Allora:
pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio_

# Cerchiamo ora le nuove coordinate degli elementi del dataset nel nuovo asse
X_trans_1 = pca_1.transform(X); X_trans_1

# Per vedere graficamente il risultato della riduzione di coordinate
X_reduced_1 = pca_1.inverse_transform(X_trans_1); X_reduced_1
plt.plot(X_reduced_1[:,0], X_reduced_1[:,1])
















