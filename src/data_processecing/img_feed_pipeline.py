import numpy as np 
from PIL import Image, UnidentifiedImageError
try:
    from .data_config import IMG_SIZE
except (ImportError, ValueError):
    from data_config import IMG_SIZE

def image_feed(img_input) -> np.ndarray:
    """
    Preprocess an input image for model prediction.
    Accepts a file path (str), a file-like object (e.g. Streamlit UploadedFile),
    or an already opened PIL Image.
    Returns:
        np.ndarray of shape (1, 256, 256, 3) normalized float array, or None on failure.
    """
    try:
        if isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        else:
            img = Image.open(img_input).convert("RGB")

        img = img.resize(IMG_SIZE)
        # Normalize pixel values
        img_arr = np.array(img, dtype=np.float32) / 255.0
        batch_img = np.expand_dims(img_arr, axis=0)
        return batch_img 

    except FileNotFoundError:
        print(f"Image not found at {img_input}")
        return None

    except UnidentifiedImageError:
        print(f"Cannot identify image from input")
        return None

    except Exception as e:
        print(f"Error in image_feed: {e}")
        return None