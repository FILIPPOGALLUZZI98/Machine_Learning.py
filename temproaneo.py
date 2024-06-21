from sklearn.model_selection import train_test_split
# Dividiamo i dati in train(80%)/ test(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
