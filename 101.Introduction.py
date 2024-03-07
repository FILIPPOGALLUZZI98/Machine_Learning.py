# Supponiamo di avere dei dati ordinati in un array
data = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# MEAN, MEDIAN, MODE
mean = np.mean(data); print(mean)
median = np.median(data); print(median)
mode = st.mode(data); print(mode)

# STANDARD DEVIATION AND VARIANCE
std = np.std(data); print(std)
var = np.var(data); print(var)

# PERCENTILES
# I percentili vengono utilizzati in statistica per fornire un numero che descrive il valore rispetto
# al quale una determinata percentuale dei valori è inferiore; X è il 75 percentile--> il 75% dei dati sono
# inferiori o uguali a X
data = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
perc_75 = numpy.percentile(data, 75); print(perc_75)

# RANDOM DATASET
# Per creare dataset grandi per prova o per testing usiamo 'random.uniform()'
# X random floats tra a and b
test = np.random.uniform(a, b, X)
# Distribuzione gaussiana con X valori, M valore medio e S standard deviation
norm = np.random.normal(M, S, X)

# HISTOGRAMS
# Supponiamo di voler costruire un istogramma dei dati 'test', con X numero di colonne
plt.hist(test, X); plt.show()

# SCATTER PLOT
# Creiamo casualmente i seguenti array di dati (con stessa lunghezza) con distribuzione normale
x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)
plt.scatter(x, y); plt.show()











