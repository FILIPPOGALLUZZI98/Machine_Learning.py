import pandas as pd
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import quandl
import math


# Una rete neurale è composta da migliaia o addirittura milioni di semplici nodi di elaborazione densamente interconnessi
# La maggior parte delle reti neurali odierne sono organizzate in strati di nodi e sono "feed-forward", il che significa 
# che i dati si muovono attraverso di esse in una sola direzione
# A ciascuna delle sue connessioni in entrata, un nodo assegnerà un numero noto come “peso”. Quando la rete è attiva, il
# nodo riceve un dato diverso – un numero diverso – su ciascuna delle sue connessioni e lo moltiplica per il peso associato
# Quindi somma insieme i prodotti risultanti, ottenendo un singolo numero. Se quel numero è inferiore a un valore soglia, il 
# nodo non passa dati al livello successivo. Se il numero supera il valore di soglia, il nodo “si accende”, il che nelle 
# odierne reti neurali generalmente significa inviare il numero – la somma degli input ponderati – lungo tutte le sue 
# connessioni in uscita
# Quando una rete neurale viene addestrata, tutti i suoi pesi e le sue soglie vengono inizialmente impostati su valori casuali
# I dati di addestramento vengono inviati allo strato inferiore, lo strato di input, e passano attraverso gli strati successivi, 
# moltiplicandosi e sommandosi in modi complessi, fino ad arrivare, radicalmente trasformati, allo strato di output

# Esistono principalmente due librerie per la creazione di reti neurali:
# tensorflow (google) e pythorch (facebook)
!pip install tensorflow #google colab
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap






























