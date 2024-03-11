# L'algoritmo di backpropagation è un supervised learning method per multilayer feed-forward networks
# Il principio della backpropagation è di modellizzare una certa funzione modificandone i pesi interni dei segnali in input
# Il training del sistema viene fatto usando un supervised learning method, dove gli errori tra l'output del sistema e quello
# conosciuto è presentato dal sistema e usato per modificareil suo stato interno
# Backpropagation può essere usato sia per classificazione che regressione

# Questo dataset contiene le previsioni di misurazioni dei semi di differenti varietà di grano
# Ci sono 201 misure e 7 variabili, è un problema di classificazione con 3 output classes


# INIZIALIZZAZIONE
# Partiamo dalla creazione di una nuova rete pronta per la formazione
# Ogni neurone ha una serie di pesi che devono essere mantenuti. Un peso per ogni connessione di ingresso e un peso aggiuntivo per
# il bias. Dovremo memorizzare proprietà aggiuntive per un neurone durante l'allenamento, quindi utilizzeremo un dizionario per 
# rappresentare ciascun neurone e memorizzare le proprietà con nomi come "pesi" per i pesi
# Una rete è organizzata in livelli. Il livello di input è in realtà solo una riga del nostro set di dati di addestramento
# Il primo vero strato è lo strato nascosto. Questo è seguito dal livello di output che ha un neurone per ogni valore di classe
# Organizzeremo i livelli come array di dizionari e tratteremo l'intera rete come un array di livelli
# È buona norma inizializzare i pesi della rete su piccoli numeri casuali: in questo caso, utilizzeremo numeri casuali compresi tra 0 e 1

# initialize_network() crea una nuova rete neurale pronta per l'addestramento. Accetta tre parametri, il numero di input, il numero di neuroni
# da avere nello strato nascosto e il numero di output
# Puoi vedere che per lo strato nascosto creiamo neuroni = n_hidden e ogni neurone nello strato nascosto ha pesi = n_inputs+1, uno per ogni
# colonna di input in un set di dati e uno aggiuntivo per il bias
# Puoi anche vedere che il livello di output che si collega al livello nascosto ha neuroni = n_outputs, ciascuno con pesi = n_hidden+1
# Ciò significa che ogni neurone nello strato di output si connette a (ha un peso per) ogni neurone nello strato nascosto
from random import seed
from random import random
from math import exp

def initialize_network(n_inputs, n_hidden, n_outputs):
 network = list()
 hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 network.append(hidden_layer)
 output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 network.append(output_layer)
 return network 
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
 print(layer)

# Possiamo ora calcolare un output da una rete neurale propagando un segnale di input attraverso ciascun livello finché
# il livello di output non restituisce i suoi valori. Chiamiamo questa propagazione in avanti
# Questa è la tecnica di cui abbiamo bisogno per generare previsioni durante il training che devono essere corrette
# Possiamo suddividere questa fase in tre parti: Neuron Activation; Neuron Transfer; Forward Propagation

# Il primo passo è calcolare l'attivazione di un neurone dato un input
# L'input potrebbe essere una riga del nostro set di dati di addestramento, come nel caso del livello nascosto. Potrebbero anche essere
# gli output di ciascun neurone nello strato nascosto, nel caso dello strato di output
# L'attivazione dei neuroni viene calcolata come la somma ponderata degli input. Proprio come la regressione lineare
activation = sum(weight_i * input_i) + bias
# Di seguito è riportata un'implementazione di ciò in una funzione denominata activate(). Puoi vedere che la funzione presuppone che
# il bias sia l'ultimo peso nell'elenco dei pesi. Questo aiuta qui e in seguito a rendere il codice più facile da leggere
def activate(weights, inputs):
 activation = weights[-1]
 for i in range(len(weights)-1):
 activation += weights[i] * inputs[i]
 return activation

# Una volta attivato un neurone, dobbiamo trasferire l'attivazione per vedere quale sia effettivamente l'output del neurone
# È possibile utilizzare diverse funzioni di trasferimento. Tradizionalmente si utilizza la funzione di attivazione del sigmoide,
# ma è anche possibile utilizzare la funzione tangente iperbolica per trasferire gli output. Più recentemente, la funzione di trasferimento
# del raddrizzatore è diventata popolare tra le grandi reti di deep learning
# La funzione di attivazione del sigmoide assomiglia a una forma a S, è anche chiamata funzione logistica. Può accettare qualsiasi valore
# di input e produrre un numero compreso tra 0 e 1 su una curva a S. È anche una funzione di cui possiamo facilmente calcolare la derivata 
# di cui avremo bisogno in seguito durante la propagazione all'indietro dell'errore
output = 1 / (1 + e^(-activation))  ## Usando la funzione sigmoide
# La propagazione in avanti di un input è semplice. Lavoriamo attraverso ogni strato della nostra rete calcolando gli output per ciascun neurone
# Tutti gli output di uno strato diventano input per i neuroni dello strato successivo.
# Di seguito è riportata una funzione denominata forward_propagate() che implementa la propagazione in avanti per una riga di dati dal nostro set
# di dati con la nostra rete neurale. Possiamo vedere che il valore di output di un neurone è memorizzato nel neurone con il nome "output".
# Possiamo anche vedere che raccogliamo gli output per un livello in un array denominato new_inputs che diventa gli input dell'array e viene
# utilizzato come input per il livello successivo
def forward_propagate(network, row):
 inputs = row
 for layer in network:
 new_inputs = []
 for neuron in layer:
 activation = activate(neuron['weights'], inputs)
 neuron['output'] = transfer(activation)
 new_inputs.append(neuron['output'])
 inputs = new_inputs
 return inputs
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
 [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)

# L'algoritmo di backpropagation prende il nome dal modo in cui vengono addestrati i pesi. L'errore viene calcolato tra le uscite 
# previste e le uscite propagate in avanti dalla rete. Questi errori vengono quindi propagati all'indietro attraverso la rete dallo
# strato di output allo strato nascosto, assegnando la colpa dell'errore e aggiornando i pesi man mano che procedono
# La matematica per l'errore di propagazione all'indietro è radicata nel calcolo, ma in questa sezione resteremo ad un livello elevato
# e ci concentreremo su cosa viene calcolato e come, piuttosto che sul perché, i calcoli assumono questa forma particolare
Questa parte è suddivisa in due sezioni: Trasferimento derivato; Errore di backpropagation.











