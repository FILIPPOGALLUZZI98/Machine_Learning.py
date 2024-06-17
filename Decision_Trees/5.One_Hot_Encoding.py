import pandas as pd

# Supponiamo di avere il dataset df
# Selezioniamo le nuove variabili
new_variables = ['var_1','var_2','var_3','var_4','var_5']

# Per fare one hot encoding con pandas
df = pd.get_dummies(data = df,prefix = new_variables,columns = new_variables)

