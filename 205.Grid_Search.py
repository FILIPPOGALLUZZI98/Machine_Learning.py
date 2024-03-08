# La maggior parte dei modelli di machine learning contiene parametri che possono essere regolati per variare il modo in cui
# il modello apprende. Ad esempio, il modello di regressione logistica, da sklearn, ha un parametro C che controlla la
# regolarizzazione, che influenza la complessità del modello
# Come scegliamo il miglior valore per C? Il valore migliore dipende dai dati utilizzati per addestrare il modello.
# Un metodo consiste nel provare valori diversi e quindi scegliere il valore che fornisce il punteggio migliore. Questa tecnica
# è nota come Grid Search. Se dovessimo selezionare i valori per due o più parametri, valuteremmo tutte le combinazioni degli
# insiemi di valori formando così una griglia di valori
# Prima di entrare nell'esempio è bene sapere cosa fa il parametro che stiamo modificando. Valori più elevati di C dicono 
# al modello, i dati di addestramento assomigliano alle informazioni del mondo reale, attribuiscono un peso maggiore ai dati
# di addestramento, mentre valori più bassi di C fanno il contrario.

# Usiamo il dataset 'idris' contenuto in sklearn
from sklearn import datasets
iris = datasets.load_iris()
# Creiamo le variabili indipendente e dipendente
X = iris['data']
y = iris['target']

# Carichiamo il modello logistico
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(max_iter = 10000)
print(logit.fit(X,y))  ## Adattiamo il modello ai dati
print(logit.score(X,y))  ## Per valutare il modello eseguiamo il metodo del punteggio

# Implementazione
# Poiché il valore predefinito per C è 1, imposteremo un intervallo di valori che lo circondano
C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# Successivamente creeremo un ciclo for per modificare i valori C e valutare il modello ad ogni modifica
# Per prima cosa creeremo un elenco vuoto in cui memorizzare il punteggio
scores = []
# Per modificare i valori di Cdobbiamo scorrere l'intervallo di valori e aggiornare il parametro ogni volta
for choice in C:
  logit.set_params(C=choice)
  logit.fit(X, y)
  scores.append(logit.score(X, y))
print(scores)
# Possiamo vedere che i valori più bassi di C hanno ottenuto risultati peggiori rispetto al parametro di base di 1
# Tuttavia, man mano che aumentavamo il valore, il C modello 1.75 riscontrava una maggiore precisione
# Sembra che aumentare C oltre questo importo non aiuti ad aumentare la precisione del modello











































