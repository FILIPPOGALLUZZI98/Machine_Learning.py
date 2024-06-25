import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Supponiamo di avere il dataset X contenente dei punti di tipo (x1, x2, x3)

# Facciamo la scomposizione in PCA
# In questo caso mettiamo le PCs uguali al numero di coordinate iniziali (in questo caso = 3)
# per vedere le diverse explained variances
pca = PCA(n_components=3)
pca.fit(X)  ## In questo caso non dobbiamo fare la normalizzazione dato che sklearn già lo fa

# Vediamo ora la explained variance per gli assi 
pca.explained_variance_ratio_  ## Vettore con 3 coordinate che sono le explained variance ratio

# Se una coordinata ha explained variance molto più grande delle altre, riduciamo le coordinate
# Supponiamo che sia (0.9, 0.05, 0.05). Allora:
pca = PCA(n_components=1)
pca.fit(X)
pca.explained_variance_ratio_

# Cerchiamo ora le nuove coordinate degli elementi del dataset nel nuovo asse
X_trans = pca.transform(X); X_trans

# Per vedere graficamente il risultato della riduzione di coordinate
X_reduced = pca.inverse_transform(X_trans); X_reduced
plt.plot(X_reduced[:,0], X_reduced[:,1])


#############################################################################################
#############################################################################################
####  EXAMPLE

# Supponiamo di avere un dataset con 1000 features e 500 campioni

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Unsupervised_Learning/'
file_path = datadir +'Principal_Components.csv'
df = pd.read_csv(file_path); df.head()

# Cerchiamo di vedere se ci sono pattern nei dati
# La seguente funzione prenderà casualmente 100 campioni di pairwise tuples (x,y) di features
def get_pairs(n = 100):
    from random import randint
    i = 0
    tuples = []
    while i < 100:
        x = df.columns[randint(0,999)]
        y = df.columns[randint(0,999)]
        while x == y or (x,y) in tuples or (y,x) in tuples:
            y = df.columns[randint(0,999)]
        tuples.append((x,y))
        i+=1
    return tuples
pairs = get_pairs()
fig, axs = plt.subplots(10,10, figsize = (35,35))
i = 0
for rows in axs:
    for ax in rows:
        ax.scatter(df[pairs[i][0]],df[pairs[i][1]], color = "#C00000")
        ax.set_xlabel(pairs[i][0])
        ax.set_ylabel(pairs[i][1])
        i+=1
# Non si nota quasi nulla

# Cerchiamo allora se ci sono correlazioni lineari (cerchiamo solo features con corr>0.5)
corr = df.corr()
mask = (abs(corr) > 0.5) & (abs(corr) != 1)
corr.where(mask).stack().sort_values()
# Anche in questo caso la correlazione massima in valore assoluto è 0.632


# Proviamo quindi la decomposizoone PCs
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(X_pca, columns = ['principal_component_1','principal_component_2'])
plt.scatter(df_pca['principal_component_1'],df_pca['principal_component_2'])
plt.xlabel('principal_component_1'); plt.ylabel('principal_component_2'); plt.title('PCA decomposition')
# Da questo plot si notano dei clusters!

# Se vogliamo vedere quanta varianza dei dati viene 'spiegata' da solo queste due componenti
sum(pca.explained_variance_ratio_)

# Aumentando le componenti si ottiene ovviamente una explained variance maggiore
pca_3 = PCA(n_components = 3).fit(df)
X_t = pca_3.transform(df)
df_pca_3 = pd.DataFrame(X_t,columns = ['principal_component_1','principal_component_2','principal_component_3'])
import plotly.express as px
fig = px.scatter_3d(df_pca_3, x = 'principal_component_1', y = 'principal_component_2', z = 'principal_component_3').update_traces(marker = dict(color = "#C00000"))
fig.show()
sum(pca_3.explained_variance_ratio_)









