import argparse
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
import keras

def load_and_summarize_model(model_path):
    # Charger le modèle
    model = keras.models.load_model(model_path, custom_objects={'tf': tf}, safe_mode=False)

    # Afficher le résumé du modèle
    model.summary()

if __name__ == "__main__":
    # Définir les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Charger et afficher le résumé d\'un modèle Keras')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle .keras')

    # Analyser les arguments
    args = parser.parse_args()

    # Appeler la fonction pour charger et afficher le modèle
    load_and_summarize_model(args.model_path)