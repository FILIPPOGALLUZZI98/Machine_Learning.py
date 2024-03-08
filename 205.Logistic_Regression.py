# La regressione logistica mira a risolvere problemi di classificazione. Lo fa prevedendo risultati categorici, a differenza
# della regressione lineare che prevede un risultato continuo
# Nel caso più semplice ci sono due risultati, detto binomio, un esempio del quale è prevedere se un tumore è maligno o benigno
# Altri casi hanno più di due risultati da classificare, in questo caso si parla di multinomiale. Un esempio comune di
# regressione logistica multinomiale sarebbe la previsione della classe di un fiore di iris tra 3 specie diverse
# Qui utilizzeremo la regressione logistica di base per prevedere una variabile binomiale, cioè che ha solo due possibili risultati
from sklearn import linear_model

# Esempio basato sui tumori
# X rappresenta la grandezza di un tumore in centimetri
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
# X deve essere formattato in una colonna 
from a row for the LogisticRegression() function to work
# y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])




















