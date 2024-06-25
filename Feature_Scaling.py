from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Supponiamo di avere i dati X, y

scaler = StandardScaler().set_output(transform="pandas")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)




