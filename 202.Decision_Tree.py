# Un decision tree è un diagramma di flusso che aiuta a prendere decisioni basate sull'esperienza precedente

# Nell'esempio, una persona proverà a decidere se andare a vedere uno spettacolo comico oppure no
# La nostra persona di esempio si è registrata ogni volta che c'era uno spettacolo comico in città, ha registrato
# alcune informazioni sullo spettacolo e ha anche registrato se era andato o meno
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv(datadir+"/data_decision_tree.csv")

# Per fare un decision tree tutti i valori devono essere numerici, quindi dobbiamo convertirli in valori numerici
# Pandas ha 'map()' che crea un dizionario con informazioni su come convertire i valori
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
print(df)

# Ora dobbiamo separare le colonne feature dalle colonne target: le prime sono le colonne da cui 
# proviamo a predire i valori, target sono i valori che cerchiamo di predire
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]; y = df['Go']
print(X); print(y)

# A questo punto possiamo creare il decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
tree.plot_tree(dtree, feature_names=features)

# La prima riga della casella, in ogni casella, indica la condizione per cui i dati si suddividono: per 
# esempio, X<=50 indica che tutti i dati con valore X minore o uguale a 50 seguiranno True gli altri False e dunque proseguiranno
# Sotto a questa condizione ci sono altre caratteristiche dei dati, per esempio indicazioni sulla qualità della
# suddivisione, quanti dati rimangono, e altro

# Ci sono molti modi per suddividere i dati, il metodo GINI usa la formula: GINI = 1 - (x/n)^2 - (y/n)^2
# Dove x è il numero di risposte positive, n è il numero di campioni, y è il numero di risposte negative

# In questo caso abbiamo, nella prima casella:
# RANK -----> Tutti gli spettacoli di rango minore o uguale a quello indicato seguiranno la True arrow
# GINI -----> Qualità della suddivisione: sempre compreso tra 0 (tutti i campioni hanno ottenuto stesso risultato)
# e 0.5 (suddivisione eseguita esattamente al centro)
# SAMPLES --> Numero di spettacoli rimanenti
# VALUE ----> Di questi samples rimanenti, indica quanti SI e quanti NO
# Il passaggio successivo contiene due caselle, una per gli spettacoli con un "Rango" pari a 6,5 ​​o inferiore e una casella con il resto
# In queste la condizione è data sulla nazionalità, poi età, poi experience
# Per spiegazione passaggio per passaggio "https://www.w3schools.com/python/python_ml_decision_tree.asp"

# Per predire nuovi valori
print(dtree.predict([[40, 10, 7, 1]]))  ## In ordine: Età, Experience, Comedy rank












