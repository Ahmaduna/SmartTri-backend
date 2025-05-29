import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore

class WasteClassifier:
    def __init__(self):
        # On ne charge plus de modèle
        self.model = None
        self.class_names = ['bleue', 'jaune', 'marron', 'noire', 'rouge', 'special', 'verte']

    def load_and_preprocess_image(self, image_file, img_size=(224, 224)):
        """
        Simule le chargement et le prétraitement de l’image.
        """
        img_bytes = image_file.read()
        img = tf.io.decode_image(img_bytes, channels=3)
        img = tf.image.resize(img, img_size)
        img = preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        return img

    def predict_image(self, image_file, confidence_threshold=0.5):
        """
        Simule une prédiction en l'absence de modèle.
        """
        try:
            # Simule une image prétraitée (non utilisée ici)
            _ = self.load_and_preprocess_image(image_file)

            # MOCK : prédiction fixe sur "jaune"
            predictions = np.array([0.1, 0.9, 0.05, 0.05, 0.01, 0.01, 0.05])  # jaune max

            confidence_scores = {self.class_names[i]: float(score) for i, score in enumerate(predictions)}
            predicted_class_index = np.argmax(predictions)
            confidence_score = float(predictions[predicted_class_index])

            if confidence_score >= confidence_threshold:
                predicted_class = self.class_names[predicted_class_index]
                bin_score = get_bin_score(predicted_class)
                product_type = get_product_type(predicted_class)

                return {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence_score': confidence_score,
                    'all_scores': confidence_scores,
                    'bin_score': bin_score,
                    'product_type': product_type,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'predicted_class': None,
                    'confidence_score': confidence_score,
                    'all_scores': confidence_scores,
                    'bin_score': None,
                    'product_type': get_product_type("unknown"),
                    'error': 'Low confidence prediction'
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

# ------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------

def get_bin_score(bin_color: str) -> int:
    """
    Convertit la couleur de la poubelle en score de recyclabilité.
    """
    scoring_system = {
        'verte': 4,
        'jaune': 2,
        'bleue': 3,
        'rouge': 6,
        'noire': 3,
        'marron': 6,
        'special': 10,
    }
    return scoring_system.get(bin_color, 1)

def get_product_type(bin_color: str) -> str:
    """
    Retourne le type de produit en fonction de la poubelle.
    """
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

# ------------------------------------------------------------------------
# INSTANCE UNIQUE
# ------------------------------------------------------------------------

waste_classifier = WasteClassifier()
