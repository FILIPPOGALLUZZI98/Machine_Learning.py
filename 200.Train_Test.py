# Per valutare se un certo modello è valido usiamo il medoto chiamato Train/test
# Dividiamo i dati in due insiemi (solitamente 80% train e 20% test)

# Creiamo una variabile indipendente random con distribuzione normale
np.random.seed(2)
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x
plt.scatter(x, y); plt.show()

# Dato che sono valori random, selezioniamo i primi 80 per train e i successivi 20 per test
train_x = x[:80]; train_y = y[:80]
test_x = x[80:]; test_y = y[80:]
plt.scatter(train_x, train_y); plt.show()
plt.scatter(test_x, test_y); plt.show()
# Vediamo che sono simili




# Proviamo una regressione polinomiale
from sklearn.metrics import r2_score 
model = np.poly1d(np.polyfit(x, y, 4))
myline = np.linspace(0, 6, 100)  ## Indichiamo la posizione di inizio e fine della linea
plt.scatter(x, y); plt.plot(myline, model(myline)); plt.show()
print(r2_score(y, model(x)))
# Probabilmente è presente un overfitting, anche se r_score è elevato

# Usiamo i dati test per vedere se si adattano bene al modello
r2 = r2_score(test_y, model(test_x)); print(r2)
# Essendo alto anche questo valore, supponiamo che il modello sia buono

# A questo punto possiamo predire i valori
print(model(5))






































