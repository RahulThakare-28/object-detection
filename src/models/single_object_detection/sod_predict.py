import logging
import os
from pathlib import Path
from tensorflow import keras

logging.basicConfig(level=logging.INFO)

_MODEL = None

def sod_predict(image):
    """
    Predict class and bounding box coordinates for a preprocessed image.
    Image must be a numpy array of shape (1, 256, 256, 3).
    """
    global _MODEL
    if _MODEL is None:
        # Support running from root or subfolders
        model_path = Path("train_models/SOD_model_v0.1.keras")
        if not model_path.exists():
            # Try looking up relative to this file
            model_path = Path(__file__).resolve().parents[3] / "train_models" / "SOD_model_v0.1.keras"
        
        logging.info(f"Loading model from {model_path}")
        _MODEL = keras.models.load_model(str(model_path), compile=False)
        
    return _MODEL.predict(image, verbose=0)
