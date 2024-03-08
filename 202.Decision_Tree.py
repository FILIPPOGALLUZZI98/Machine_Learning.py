# Un decision tree è un diagramma di flusso che aiuta a prendere decisioni basate sull'esperienza precedente
# Nell'esempio, una persona proverà a decidere se andare a vedere uno spettacolo comico oppure no
# La nostra persona di esempio si è registrata ogni volta che c'era uno spettacolo comico in città, ha registrato
# alcune informazioni sul comico e ha anche registrato se era andato o meno
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv(datadir+"/data_decision_tree.csv")

# Per fare un decision tree tutti i valori devono essere numerici, quindi dobbiamo convertirli
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



















