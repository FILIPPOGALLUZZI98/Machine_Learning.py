import pandas as pd
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import quandl
import math

df = quandl.get("WIKI/GOOGL")
print(df.head())
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# Non tutti i dati di cui disponi sono utili e talvolta è necessario eseguire ulteriori manipolazioni sui dati 
# per renderli ancora più preziosi prima di inserirli in un algoritmo di apprendimento automatico
# Andiamo avanti e trasformiamo i nostri dati successivamente:
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())


































