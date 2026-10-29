
import numpy as np 
from PIL import UnidentifiedImageError
from PIL import Image 
from data_config import IMG_SIZE 

def image_feed(img_path:str)-> np.ndarray :
    try :
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img = np.array(img)/225.0
        batch_img = np.expand_dims(img, axis=0)
        return batch_img 

    except FileNotFoundError:
        print(f"Image not found at {img_path}")
        return None

    except UnidentifiedImageError:
        print(f"Cannot identify image at {img_path}")
        return None