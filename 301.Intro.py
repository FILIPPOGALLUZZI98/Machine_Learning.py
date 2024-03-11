# In questo primo esempio, cerchiamo di classificare delle specie di iris trovate
# Supponiamo di avere misurazioni di lunghezza e larghezza dei petali, e della lunghezza e larghezza dei sepali (in cm)
# Supponiamo anche di avere misure già identificate da un botanico: setosa, versicolor, virginica
# Siccome abbiamo misure di cui siamo certi della specie, si tratta di un supervised learning problem
# I possibili output sono chiamati classi; per un particolare data point, la specie a cui appartiene è chimaata label
# Gli oggetti individuali sono chiamati samples e le loro caratteristiche features

from sklearn.datasets import load_iris
iris_dataset = load_iris()
# Il dataset è simile a un dizionario, quindi cerchiamo i keys e values
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# Per mostrare il contenuto delle varie keys
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
# Vediamo che l'array contiene misure per 105 fiori differenti
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# Da questo possiamo vedere che tutti i primi 5 fiori hanno petali di larghezza 0.2cm e che il primo
# fiore ha sepalo di 5.1cm




# Vogliamo costruire un modello ML da questi dati che ci permetta di predire la specie di iris di una nuova misurazione
# Prima di fare ciò è necessario suddividere il dataset in due parti (train/test)
# Possiamo farlo con la funzione 'train_test_split' che divide il dataset in 75-25
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

# Siccome vogliamo vedere il dataset graficamente, ma continene più di due variabili, usiamo il pair plot,
# che considera tutte le coppie possibili delle variabili
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# Dai grafici si osserva che le classi sembrano relativamente ben separate usando sepal e petal measurements

# In questo caso usiamo il metodo k-nearest neighbors classifier (k significa che, invece di considerare solo
# il più vicino, consideriamo un numerofisso k di vicini)
# Tutti i modelli ML in scikit-learn sono implementati nelle loro classi, chiamate estimator classes
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)  ## Impostiamo il numero di vicini uguale a 1
# L'oggetto knn contiene l'algorirmo che verrà usato per costruire il modello dal training data, oltre a fare
# predizioni sui nuovi dati
# Per costruire il modello sul training set, applichiamo il metodo fit all'oggetto knn
knn.fit(X_train, y_train)
# Il metodo fit restituisce l'oggetto knn stesso, quindi abbiamo una rappresentazione di stringa del classifier
CONTINUA DA PAGINA 22










