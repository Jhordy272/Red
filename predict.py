import numpy as np
import tensorflow as tf

# Cargar el modelo guardado
modelo_cargado = tf.keras.models.load_model("Dieta.h5")

# Ejemplo de datos de entrada
# Supongamos que tus características son: Edad, Sexo, Peso, Altura, Actividad Física, Objetivo, Restricciones, Preferencia
datos_entrada = np.array([[25, 0, 70, 175, 2, 1, 0, 0]])  # Reemplaza estos valores con los que quieras predecir

# Realizar la predicción
prediccion = modelo_cargado.predict(datos_entrada)

# Mostrar la predicción
print("Predicción (Proteínas, Carbohidratos, Grasas, Calorías):", prediccion)