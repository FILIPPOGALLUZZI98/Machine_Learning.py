# Se abbiamo dei dati categorici non possiamo usarli nei metodi di machine learning
# Per ovviare al problema, dobbiamo fornire una rappresentazione numerica delle variabili categoriche
# Un modo di farlo è avere una colonna che rappresenta ogni gruppo nella categoria
# Per ogni colonna, i valori saranno 1 o 0, dove 1 rappresenta l'inclusione, 0 l'esclusione: chiamato hot encoding
# pandas ha un modulo che si chiama get_dummies()
from sklearn import linear_model

# Usiamo i dati delle macchine viste nel file '102'
cars = pd.read_csv(datadir+"/data_multiple_regr.csv")
# Con questo creaiamo una colonna per ogni marca di macchine
ohe_cars = pd.get_dummies(cars[['Car']])
print(ohe_cars.to_string())

# A questo punto dobbiamo selezionare una variabile indipendente X ed aggiungere una colonna dei dummy variables
# Anche aggiungere la variabile dipendente in y
X = pandas.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
y = cars['CO2']

# Usiamo ora la regressione linerae
regr = linear_model.LinearRegression()
regr.fit(X,y)

# A questo punto possiamo predire le emissioni di CO2 basate su peso, volume e marca
predictedCO2 = regr.predict([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])  ## Marca n.17















