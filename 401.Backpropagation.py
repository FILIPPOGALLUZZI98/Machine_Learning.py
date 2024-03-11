# L'algoritmo di backpropagation è un supervised learning method per multilayer feed-forward networks
# Il principio della backpropagation è di modellizzare una certa funzione modificandone i pesi interni dei segnali in input
# Il training del sistema viene fatto usando un supervised learning method, dove gli errori tra l'output del sistema e quello
# conosciuto è presentato dal sistema e usato per modificareil suo stato interno

# Backpropagation può essere usato sia per classificazione che regressione
# Questo dataset contiene le previsioni di misurazioni dei semi di differenti varietà di grano
# Ci sono 201 misure e 7 variabili, è un problema di classificazione con 3 output classes


# INIZIALIZZAZIONE
# Partiamo da qualcosa di semplice, la creazione di una nuova rete pronta per la formazione
# Ogni neurone ha una serie di pesi che devono essere mantenuti. Un peso per ogni connessione di ingresso e un peso aggiuntivo per
# il bias. Dovremo memorizzare proprietà aggiuntive per un neurone durante l'allenamento, quindi utilizzeremo un dizionario per 
# rappresentare ciascun neurone e memorizzare le proprietà con nomi come "pesi" per i pesi
# Una rete è organizzata in livelli. Il livello di input è in realtà solo una riga del nostro set di dati di addestramento
# Il primo vero strato è lo strato nascosto. Questo è seguito dal livello di output che ha un neurone per ogni valore di classe
# Organizzeremo i livelli come array di dizionari e tratteremo l'intera rete come un array di livelli
# È buona norma inizializzare i pesi della rete su piccoli numeri casuali. In questo caso, utilizzeremo numeri casuali compresi tra 0 e 1

# initialize_network() crea una nuova rete neurale pronta per l'addestramento. Accetta tre parametri, il numero di input, il numero di neuroni
# da avere nello strato nascosto e il numero di output
# Puoi vedere che per lo strato nascosto creiamo n_hidden neuroni e ogni neurone nello strato nascosto ha pesi = n_inputs+1, uno per ogni
# colonna di input in un set di dati e uno aggiuntivo per il bias
# Puoi anche vedere che il livello di output che si collega al livello nascosto ha neuroni = n_outputs, ciascuno con pesi=n_hidden+1
# Ciò significa che ogni neurone nello strato di output si connette a (ha un peso per) ogni neurone nello strato nascosto
def initialize_network(n_inputs, n_hidden, n_outputs):
 network = list()
 hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 network.append(hidden_layer)
 output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 network.append(output_layer)
 return network





















