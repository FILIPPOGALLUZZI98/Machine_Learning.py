# Questo algoritmo viene usato per ottimizzare la scelta dei parametri alfa negli
# algoritmi di minimizzazione della loss function
# Consideriamo il modello improved di multiclass classification

# Il modello rimane lo stesso ma
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss = tf.keras-losses.SparseCategoricalCrossentropy(from_logits=True))

# Anche il fit rimane uguale
model.fit(X,Y,epochs=100)
