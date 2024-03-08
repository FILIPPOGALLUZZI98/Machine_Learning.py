# Il clustering gerarchico è un metodo di apprendimento non supervisionato per il clustering di punti dati
# L'algoritmo crea cluster misurando le differenze tra i dati. L'apprendimento non supervisionato significa che
# un modello non deve essere addestrato e non abbiamo bisogno di una variabile "target". Questo metodo può essere utilizzato
# su qualsiasi dato per visualizzare e interpretare la relazione tra i singoli punti dati

# Utilizzeremo l'Agglomerative Clustering, un tipo di clustering gerarchico che segue un approccio down-top
# Iniziamo trattando ciascun punto dati come un proprio cluster. Quindi, uniamo insieme i cluster che hanno la distanza più
# breve tra loro per creare cluster più grandi. Questo passaggio viene ripetuto finché non viene formato un cluster di grandi
# dimensioni contenente tutti i punti dati
# Il clustering gerarchico ci impone di decidere sia il metodo della distanza che quello del collegamento. Utilizzeremo la distanza
# euclidea e il metodo del collegamento di Ward, che tenta di minimizzare la varianza tra i cluster

from scipy.cluster.hierarchy import dendrogram, linkage
# Iniziamo plottando i dati
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))  ## Questo serve a unire i dati in un insieme di punti (x,y)
plt.scatter(x, y); plt.show()

# Ora calcoliamo il collegamento dei cluster utilizzando la distanza euclidea e lo visualizziamo utilizzando un dendrogramma
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()
# Questo grafico mostra la gerarchia dei cluster dal basso (individual points) all'alto (singolo cluster contenente tutti i dati)

# Con la libreria scikit-learn possiamo fare anche in un altro modo
from sklearn.cluster import AgglomerativeClustering
# Innanzitutto, inizializziamo la AgglomerativeClustering class con 2 cluster, utilizzando la stessa distanza euclidea e il collegamento di Ward
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# Il '.fit_predict' può essere richiamato sui nostri dati per calcolare i cluster utilizzando i parametri definiti nel numero di cluster scelto
labels = hierarchical_cluster.fit_predict(data)
plt.scatter(x, y, c=labels); plt.show()

















