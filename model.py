# Importar TensorFlow y pandas
import tensorflow as tf
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("dataset.csv")

# Separar en características (X) y etiquetas (y)
X_train = dataset.iloc[:, :8].values  # Columnas de entrada
y_train = dataset.iloc[:, 8:].values  # Columnas de salida (Proteínas, Carbohidratos, Grasas, Calorías)

# Definir el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=8, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')  # 4 salidas para proteínas, carbohidratos, grasas, calorías
])

# Compilar el modelo con una pérdida de regresión
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Usar pérdida para regresión
              metrics=['mae'])  # "mae" para el error absoluto medio

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
model.save("Dieta.h5")