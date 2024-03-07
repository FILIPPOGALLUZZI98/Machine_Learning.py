# Quando i dati hanno diverse unità di grandezza, compararli è difficile
# Per questo dobbiamo usare lo scaling
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



