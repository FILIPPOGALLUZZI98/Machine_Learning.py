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
# X deve essere formattato in una colonna da una riga per 'LogisticRegression()' 
# y rappresenta se un tumore è maligno o no (0 per no, 1 per si)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# LogisticRegression() ha un metodo chiamato 'fit()' che prende valori indipendenti e dipendenti come parametri e riempie l'oggetto 
# di regressione con i dati che descrivono la relazione
logr = linear_model.LogisticRegression()
logr.fit(X,y)

# Per predire se un tumore è maligno o meno
predicted = logr.predict(np.array([3.46]).reshape(-1,1))

# I coefficienti ci danno molte informazioni
log_odds = logr.coef_
odds = np.exp(log_odds); odds
# Per esempio in questo caso ci dice che se il tumore aumenta di 1mm, la sua probabilità di esseree maligno è 4x

# Il coefficiente e l'intercetta possono essere usati per trovare la probabilità che ciascun tumore sia maligno
def logit2prob(logr,x):
  log_odds = logr.coef_ * x + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)
print(logit2prob(logr, X))













