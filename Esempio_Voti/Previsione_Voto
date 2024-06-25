#############################################################################################
#############################################################################################
####  UN SOLO PARTITO

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Utilizzeremo un modello di classificazione come la Logistic Regression per predire se un elettore
# voterà per il Partito A.

# Caricare il dataset
df = pd.read_csv(datadir+'Esempi_Per_Elettori/'+'voters_data.csv')
X = df[['age', 'education_level', 'previous_vote', 'social_media_activity']]; y = df['voted_party']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializzare e addestrare il modello
model = LogisticRegression()
model.fit(X_train, y_train)

# Fare previsioni sul test set
y_pred = model.predict(X_test)

# Valutare l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy}')
# Mostrare la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matrice di Confusione:')
print(conf_matrix)
# Mostrare il rapporto di classificazione
class_report = classification_report(y_test, y_pred)
print('Rapporto di Classificazione:')
print(class_report)

# Predire la probabilità che un elettore voti per il Partito A
y_prob = model.predict_proba(X_test)[:, 1]

# Aggiungere le probabilità al dataframe di test
X_test['voted_party_prob'] = y_prob

# Identificare gli elettori indecisi (probabilità vicina a 0.5)
indecisive_voters = X_test[(y_prob > 0.4) & (y_prob < 0.6)]
print('Elettori Indecisi:')
print(indecisive_voters)


#############################################################################################
#############################################################################################
####  USANDO RETE NEURALE PER PIU PARTITI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

dataset = pd.read_csv(datadir+'Esempi_Per_Elettori/'+'voters_data_multiclass.csv')

# Supponiamo che l'ultima colonna sia l'etichetta (0 = non persuadibile, 1 = persuadibile)
X = dataset.iloc[:, :-1]; y = dataset.iloc[:, -1]

# Codifica delle variabili categoriali (es. partito votato)
encoder = OneHotEncoder()
partito_votato_encoded = encoder.fit_transform(X[['voted_party']]).toarray()

# Rimpiazziamo la colonna originale con le nuove colonne codificate
X = X.drop(columns=['voted_party'])
X = pd.concat([X, pd.DataFrame(partito_votato_encoded)], axis=1)

# Normalizzazione dei dati
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)

# 3. Divisione dei Dati in Training e Test Set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Costruzione del Modello di Rete Neurale
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 5. Compilazione del Modello
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Addestramento del Modello
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# 7. Valutazione del Modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")




