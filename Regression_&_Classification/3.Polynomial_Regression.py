import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Supponiamo di dover prevedere la resa di un campo in base ad alcune variabili come
# la quantità di fertilizzanti usati, di acqua e ore di sole
# Usiamo solo queste 3 variabili (anche se nel dataset ce ne sono di più)


#############################################################################################
#############################################################################################
####  DATASET

# Apriamo il dataset (supponiamo di aver già montato gdrive su colab)
datadir = datadir + 'Regression_&_Classification/'
file_path = datadir +'Polynomial_Regression_Multiple_Crop.csv'
df = pd.read_csv(file_path)

# Plottiamo i 3 grafici (fertilizer, water, sunlight vs. crop yield)
features = ['fertilizer_used','water_irrigated', 'sunlight_hours']
titles = ['Fertilzer', 'Water','Sunlight']
plt.figure(figsize=(12,10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['crop_yield'], s = 1, color = 'k')
    plt.title(titles[i]); plt.ylabel('Crop Yield')
plt.tight_layout(); plt.show()


#############################################################################################
#############################################################################################
####  MODEL SELECTION

# Data split in 3: train, cross-validation e test 
X = df[['fertilizer_used', 'water_irrigated', 'sunlight_hours']].values
y = df['crop_yield'].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  ## 60-40
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  ## 50-50

Lambda = 0.5  ## Parametro di regolarizzazione
degrees = range(1, 11)  ## Gradi del polinomio da testare
mse_train_list = []; mse_cv_list = []; mse_test_list = [] 

# Testiamo diversi gradi del polinomio
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=Lambda))
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_train_list.append(mse_train)
    cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_cv = -np.mean(cv_scores)
    mse_cv_list.append(mse_cv)
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mse_test_list.append(mse_test)
    print(f'Degree {degree}: Train MSE = {mse_train}, CV MSE = {mse_cv}, Test MSE = {mse_test}')

# Visualizziamo gli MSE per ogni grado del polinomio
plt.figure(figsize=(10, 10))
plt.plot(degrees, mse_train_list, marker='o', linestyle='-', color='b', label='Train MSE')
plt.plot(degrees, mse_cv_list, marker='o', linestyle='-', color='g', label='CV MSE')
plt.plot(degrees, mse_test_list, marker='o', linestyle='-', color='r', label='Test MSE')
plt.title('MSE per Grado del Polinomio'); plt.xlabel('Grado del Polinomio')
plt.ylabel('Mean Squared Error (MSE)'); plt.xticks(degrees); plt.legend()
plt.grid(True); plt.show()


#############################################################################################
#############################################################################################
####  MODEL IMPLEMENTATION

# Creiamo il modello polinomiale con i migliori valori trovati
degree = 3  ## Grado del polinomio
Lambda = 0.5  ## Parametro di regolarizzazione

# Applichiamo il modello
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=Lambda))
model.fit(X_train, y_train)

# Creiamo i grafici delle linee di regressione
plt.figure(figsize=(14, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[feature], df['crop_yield'], s=1, color='k', label='Dati')
    x_vals = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    X_vals = np.full((x_vals.shape[0], X.shape[1]), df[features].mean(axis=0))
    X_vals[:, i] = x_vals.flatten()
    y_vals = model.predict(X_vals)
    plt.plot(x_vals, y_vals, color='r', label='Regressione Polinomiale')
    plt.title(titles[i]); plt.xlabel(feature); plt.ylabel('Crop Yield'); plt.legend()
plt.tight_layout(); plt.show()

# Per ottenere la formula del modello
polynomial_features = model.named_steps['polynomialfeatures']
ridge = model.named_steps['ridge']
feature_names = polynomial_features.get_feature_names_out(features)
coefficients = ridge.coef_; intercept = ridge.intercept_
terms = [f"{coeff:.4f}*{name}" for coeff, name in zip(coefficients, feature_names)]
formula = " + ".join(terms); formula = f"{intercept:.4f} + " + formula
print("La forma matematica del modello è:"); print(formula)
















