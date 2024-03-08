# Si tratta di una tabella utilizzata nei problemi di classificazione per valutare dove sono stati commessi errori nel modello
# Le righe rappresentano le classi effettive in cui avrebbero dovuto essere i risultati, mentre le colonne rappresentano le previsioni
# che abbiamo fatto. Utilizzando questa tabella è facile vedere quali previsioni sono errate
from sklearn import metrics

# Generiamo valori reali e predetti 
actual = np.random.binomial(1, 0.9, size = 1000)
predicted = np.random.binomial(1, 0.9, size = 1000)

# Creiamo la confusion matrix
conf_mat = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(conf_mat, display_labels = [False, True])
cm_display.plot(); plt.show()

# True significa che i valori sono stati previsti accuratamente, False significa che si è verificato un errore o una previsione errata
# Ci dice quindi quanti falsi positivi, falsi negativi, veri negativi e veri positivi sono stati calcolati dal modello


# METRICS
# La matrice ci fornisce molte metriche utili che ci aiutano a valutare il nostro modello di classificazione:

# ACCURACY
# Accuracy misura quanto spesso il modello è corretto
# (True Positive + True Negative) / Total Predictions
Accuracy = metrics.accuracy_score(actual, predicted)

# PRECISION
# Dei valori predetti positivi, quale percentuale è veramente positiva?
# True Positive / (True Positive + False Positive)
# La precisione non valuta i casi negativi correttamente previsti
Precision = metrics.precision_score(actual, predicted)

# SENSITIVITY
# Misura la capacità del modello di prevedere i risultati positivi
# Ciò significa che esamina i veri positivi e i falsi negativi (che sono positivi che sono stati erroneamente previsti come negativi)
# True Positive / (True Positive + False Negative)
Sensitivity_recall = metrics.recall_score(actual, predicted)

# SPECIFICITY
# Quanto è efficace il modello nel prevedere i risultati negativi?
# La specificity è simile alla sensibility, ma la considera dalla prospettiva dei risultati negativi
# True Negative / (True Negative + False Positive)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)

# F-SCORE
# F-Score è la "media armonica" di precisione e sensibilità
# Considera sia i casi falsi positivi che quelli falsi negativi ed è utile per set di dati sbilanciati
# 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
F1_score = metrics.f1_score(actual, predicted)

# TUTTE LE METRICS INSIEME 
Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Sensitivity_recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F1_score = metrics.f1_score(actual, predicted)
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})













