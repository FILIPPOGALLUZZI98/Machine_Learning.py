## STATISTICS BASICS
mean = np.mean(data); print(mean)
median = np.median(data); print(median)
mode = st.mode(data); print(mode)
std = np.std(data); print(std)
var = np.var(data); print(var)
perc_75 = numpy.percentile(data, 75); print(perc_75)

# Random Datasets
test = np.random.uniform(a, b, X)
# Distribuzione Gaussiana
norm = np.random.normal(M, S, X)

#################################################################################################
#################################################################################################
## LINEAR REGRESSION
# Supponiamo di avere i seguenti array di dati (di uguale lunghezza)
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Per tracciare la retta di regressione
slope, intercept, r, p, std_err = st.linregress(x, y)
print(r)
def reg_lin(x):
  return slope * x + intercept
model = list(map(reg_lin, x))  ## This will result in a new array with new values for the y-axis
plt.scatter(x, y); plt.plot(x, model); plt.show()

# Previsione
predict = reg_lin(10); print(predict)




## POLYNOMIAL REGRESSION
from sklearn.metrics import r2_score 
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

model = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(1, 22, 100)  ## Indichiamo la posizione di inizio e fine della linea
plt.scatter(x, y); plt.plot(model, model(myline)); plt.show()
print(r2_score(y, poly(x)))

# Previsione
predict = model(17); print(predict)




## MULTIPLE REGRESSION
# Il file si può trovare qui "https://www.w3schools.com/python/data.csv"
df = pd.read_csv(datadir+"/data_multiple_regr.csv"); df
X = df[['Weight', 'Volume']]; y = df['CO2']
# Quindi in questo caso avremo la quantità di CO2 in base a peso e volume
from sklearn import linear_model as lm

regr = lm.LinearRegression(); regr.fit(X, y)
print(regr.coef_)
# I coefficienti ci dicono quando aumenta la variabile dipendente se la variabile
# indipendente aumenta di una unità

# Per predire valori di CO2 in base a peso e volume:
predCO2 = regr.predict([[2300, 1300]])

#################################################################################################
#################################################################################################
## SCALE
# Quando i dati hanno diverse unità di grandezza, compararli è difficile. Per questo usiamo lo scaling
# Il metodo di standardizzazione più usato è
z = (x-u)/s  ## x è il valore, u è la media, s è la deviazione standard

# Esiste anche un pacchetto che rende il processo più veloce
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(datadir+"/data_multiple_regr.csv")
scale = StandardScaler()
X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)
print(scaledX)
# Abbiamo riscalato Peso e Volume in grandezze che possono essere comparate




# Quando il dataset è riscalato, per fare regressione bisogna usare scale
from sklearn import linear_model as lm
regr = lm.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2300, 1.3]])
predCO2 = regr.predict([scaled[0]]); print(predCO2)

#################################################################################################
#################################################################################################
## TRAIN/TEST
# AGGIUNGERE MODO MIGLIORE PER SELEZIONARE DATI RANDOM
# Creiamo una variabile indipendente random con distribuzione normale
np.random.seed(2)
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

train_x = x[:80]; train_y = y[:80]
test_x = x[80:]; test_y = y[80:]
plt.scatter(train_x, train_y); plt.show()
plt.scatter(test_x, test_y); plt.show()
# Vediamo che sono simili




# Per vedere se funziona facciamo una regressione polinomiale
from sklearn.metrics import r2_score 
model = np.poly1d(np.polyfit(x, y, 4))
myline = np.linspace(0, 6, 100)  ## Indichiamo la posizione di inizio e fine della linea
plt.scatter(x, y); plt.plot(myline, model(myline)); plt.show()
print(r2_score(y, model(x)))
# Probabilmente è presente un overfitting, anche se r_score è elevato

# Usiamo i dati test per vedere se si adattano bene al modello
r2 = r2_score(test_y, model(test_x)); print(r2)
# Essendo alto anche questo valore, supponiamo che il modello sia buono

# Previsione
print(model(5))













