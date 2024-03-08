# K-means è un metodo di apprendimento non supervisionato per il clustering di punti dati. L'algoritmo divide
# iterativamente i punti dati in K cluster riducendo al minimo la varianza in ciascun cluster
# Ciascun punto dati viene assegnato in modo casuale a uno dei K cluster. Quindi, calcoliamo il centroide 
# (funzionalmente il centro) di ciascun cluster e riassegniamo ciascun punto dati al cluster con il centroide più
# vicino. Ripetiamo questo processo finché le assegnazioni dei cluster per ciascun punto dati non cambiano più
# K-significa che il clustering richiede di selezionare K, il numero di cluster in cui vogliamo raggruppare i dati
# Il metodo del gomito ci consente di rappresentare graficamente l'inerzia (una metrica basata sulla distanza) e visualizzare
# il punto in cui inizia a diminuire linearmente. Questo punto è denominato "eblow" ed è una buona stima del miglior valore di K in base ai nostri dati
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Dal metodo elbow vediamo che 2 è un buon valore per K, quindi:
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_); plt.show()


























