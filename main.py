


import numpy as np 
from src.data_processecing.data_config import ANNOTATION_DIR
from src.data_processecing.process_rawdata import dataOrganize










if __name__ == "__main__": 

    print("Data Processing started....")
    images, labels, bboxes = dataOrganize(ANNOTATION_DIR)
    print("Data processed successfully")

    print("list to array conversion started....")
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int8)
    bboxes = np.array(bboxes, dtype=np.float32)
    print("list to array conversion completed successfully")

    print("save processed data")
    np.save("data/process-data/processImages", images)
    np.save("data/process-data/processLabels", labels)
    np.save("data/process-data/processBBoxes", bboxes)
    print("save processed data completed successfully")

    print("Process Data Details :")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"BBoxes shape: {bboxes.shape}")
    

    