# In questo primo esempio, cerchiamo di classificare delle specie di iris trovate
# Supponiamo di avere misurazioni di lunghezza e larghezza dei petali, e della lunghezza e larghezza dei sepali (in cm)
# Supponiamo anche di avere misure già identificate da un botanico: setosa, versicolor, virginica
# Siccome abbiamo misure di cui siamo certi della specie, si tratta di un supervised learning problem
# I possibili output sono chiamati classi

from sklearn.datasets import load_iris
iris_dataset = load_iris()
# Il dataset è simile a un dizionario, quindi cerchiamo i keys e values
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# DESCR è una piccola descrizione del dataset






