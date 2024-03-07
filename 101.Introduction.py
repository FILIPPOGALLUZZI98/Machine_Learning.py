# Supponiamo di avere dei dati ordinati in un array
data = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# MEAN, MEDIAN, MODE
mean = np.mean(data)
median = np.median(data)
mode = st.mode(data)

# STANDARD DEVIATION AND VARIANCE
std = np.std(data)
var = np.var(data)

# PERCENTILES
# I percentili vengono utilizzati in statistica per fornire un numero che descrive il valore rispetto
# al quale una determinata percentuale dei valori è inferiore; X è il 75 percentile--> il 75% dei dati sono
# inferiori o uguali a X
data = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
perc_75 = numpy.percentile(data, 75)

# Per creare dataset grandi per prova o per testing usiamo 'random.uniform()'
# X random floats tra a and b
test = np.random.uniform(a, b, X)















