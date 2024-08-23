import numpy as np
import tensorflow as tf

# Chargement du modèle TFLite
interpreter = tf.lite.Interpreter(model_path="/Users/ambrericouard/sdsandbox/mycar/models/pilot_24-06-28_2.tflite")
interpreter.allocate_tensors()

# Vérification des détails du modèle TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Exemple d'utilisation du modèle TFLite avec des données d'entrée
input_shape = input_details[0]['shape']
input_data = np.ones(input_shape, dtype=np.float32)  # Exemple de données d'entrée, assurez-vous qu'elles correspondent aux spécifications attendues

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Récupération des résultats
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output data:", output_data)