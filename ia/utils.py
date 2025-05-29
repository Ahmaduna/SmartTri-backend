import numpy as np
import tensorflow as tf  # Facultatif ici, juste pour garder les dépendances si nécessaire
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
import os
from django.conf import settings

class WasteClassifier:
    def __init__(self):
        # Ne pas charger de modèle
        print("⚠️ Aucun modèle chargé – les prédictions seront simulées.")
        
        # Définir les classes comme avant
        self.class_names = ['bleue', 'jaune', 'marron', 'noire', 'rouge', 'special', 'verte']

    def load_and_preprocess_image(self, image_file, img_size=(224, 224)):
        """
        Simule la lecture d'image – pas utile sans le modèle, mais garde la signature.
        """
        return None  # Pas besoin de traitement réel

    def predict_image(self, image_file, confidence_threshold=2):
        """
        Simule une prédiction factice et retourne une classe aléatoire.
        """
        try:
            # Simulation d'une classe prédite
            predicted_class_index = np.random.randint(len(self.class_names))
            predicted_class = self.class_names[predicted_class_index]
            confidence_score = float(np.round(np.random.uniform(2.1, 5.0), 2))  # Toujours au-dessus du seuil
            confidence_scores = {
                cls: float(np.round(np.random.uniform(0.1, 5.0), 2))
                for cls in self.class_names
            }

            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence_score': confidence_score,
                'all_scores': confidence_scores,
                'bin_score': get_bin_score(predicted_class),
                'product_type': get_product_type(predicted_class),
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'predicted_class': None,
                'confidence_score': None,
                'all_scores': None,
                'bin_score': None,
                'product_type': get_product_type("unknown"),
                'error': str(e)
            }

##################################################################
# HELPER FUNCTIONS
##################################################################
def get_bin_score(bin_color: str) -> int:
    """
    Convert bin color to a numerical score based on recyclability.
    """
    scoring_system = {
        'verte': 4,    # Glass is highly recyclable
        'jaune': 2,    # Plastic/cardboard
        'bleue': 3,    # Paper
        'rouge': 6,    # Metal
        'noire': 3,    # General waste
        'marron': 6,   # Organic waste
        'special': 10, # Special/hazardous
    }
    return scoring_system.get(bin_color, 1)

def get_product_type(bin_color: str) -> str:
    if bin_color == "unknown":
        return "Unknown"
    product_types = {
        'verte': "Glass Product",
        'jaune': "Mixed Recyclable Product",
        'bleue': "Paper Product",
        'rouge': "Metal Product",
        'noire': "Non-Recyclable Product",
        'marron': "Organic Product",
        'special': "Hazardous/Special Product"
    }
    return product_types.get(bin_color, "Unknown")

# Créer une instance unique du classificateur
waste_classifier = WasteClassifier()
