


from .data_config import CLASS_MAP, IMAGE_SIZE

from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET

def dataOrganize(ANNOTATION_DIR): 

    class_labels = [] 
    images = []
    bboxes = []
    
    for xmlFile in os.listdir(ANNOTATION_DIR):
        # only .xml file
        if not xmlFile.endswith(".xml"):
            continue

        # find root of xml file too read it content
        xml_path = os.path.join(ANNOTATION_DIR, xmlFile) 
        root = ET.parse(xml_path).getroot()


        # get all class label and map with class int id
        label = root.find("object/name").text
        class_id = CLASS_MAP.get(label)
        class_labels.append(class_id)

        # get image and convert into ndarray
        filename = root.find("filename").text
        image_path = os.path.join(ANNOTATION_DIR, filename)
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image = np.array(image) / 225.0 # normlize pixel values in 0 to 1
        images.append(image)

        # get bouding boxes
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        xmin = int(root.find("object/bndbox/xmin").text)
        ymin = int(root.find("object/bndbox/ymin").text)
        xmax = int(root.find("object/bndbox/xmax").text)
        ymax = int(root.find("object/bndbox/ymax").text)

        bbox = [xmin/width, ymin/height, xmax/width, ymax/height]
        bboxes.append(bbox)


    return images, class_labels, bboxes

