
import logging
logging.basicConfig(level=logging.INFO)

from sklearn.model_selection import train_test_split
import numpy as np

from sod_model import sod_modelv01


def train_model():
    logging.info("Loading preprocess data")
    x = np.load("../data/process-data/processImages.npy")
    yc = np.load("../data/process-data/processLabels.npy")
    ybb = np.load("../data/process-data/processBBoxes.npy")
    logging.info("Data loaded successfully")

    logging.info("Spliting data into training and testing sets")
    x_train, x_test, yc_train, yc_test, ybb_train, ybb_test = train_test_split(x, yc, ybb, test_size=0.25, random_state=42)
    logging.info("Data split successfully")

    logging.info("loading model")
    model = sod_modelv01()
    logging.info("model loaded successfully")

    logging.info("training model")
    history = model.fit(
        x_train,
        {
            'class_output':yc_train,
            'bbox_output':ybb_train
        },
        epochs=1,
        batch_size=32,
        validation_split=0.2
    )

    logging.info("model trained successfully")
    logging.info("saving model")
    model.save("../models/sod_model_v0.1.keras")
    logging.info("model saved successfully")

if __name__ == "__main__":
    train_model()
