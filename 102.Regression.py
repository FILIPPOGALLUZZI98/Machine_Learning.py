# LINEAR REGRESSION
# Supponiamo di avere i seguenti array di dati (di uguale lunghezza)
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y); plt.show()

# Per tracciare la retta di regressione
slope, intercept, r, p, std_err = st.linregress(x, y)
print(r)
def reg_lin(x):
  return slope * x + intercept
mymodel = list(map(reg_lin, x))  ## This will result in a new array with new values for the y-axis
plt.scatter(x, y); plt.plot(x, mymodel); plt.show()

# Con la funzione creata possiamo anche predire i valori dei dati che non abbiamo
predict = reg_lin(10); print(predict)




# POLYNOMIAL REGRESSION
from sklearn.metrics import r2_score 
# Partiamo con due dataset
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x, y); plt.show()

poly = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(1, 22, 100)  ## Indichiamo la posizione di inizio e fine della linea
plt.scatter(x, y); plt.plot(myline, poly(my  line)); plt.show()
print(r2_score(y, poly(x)))

# Anche in questo caso possiamo predire i valori dei dati che non abbiamo
predict = poly(17); print(predict)




# MULTIPLE REGRESSION
# Regressione multipla è come regressione lineare ma con piùdi una variabile indipendente
# Il file si può trovare qui "https://www.w3schools.com/python/data.csv"
# Bisogna inserire la directory in cui è presente il file
df = pd.read_csv("data.csv")














