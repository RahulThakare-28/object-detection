
import logging
logging.basicConfig(level=logging.INFO)

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import VGG16


base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base.trainable = False  # Keep backbone frozen initially


def sod_modelv01():
    """
     Single Object Detection (SOD) model, it detect one object in the image, predict class and bouding box
    """
    logging.info("Building SOD model v0.1")
    input_layer = layers.Input(shape=(256, 256, 3))

    # vgg16 base model for feature extraction from input image
    base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base.trainable = False  # Keep backbone frozen initially
    backbone = base(input_layer)

    # global average pooling to reduce the number of features
    features = layers.GlobalAveragePooling2D()(backbone)

    # --- CLASSIFICATION HEAD (with Regularization) ---
    class_head = layers.Dense(units=256, activation='relu')(features)
    class_head = layers.BatchNormalization()(class_head)
    class_head = layers.Dropout(0.4)(class_head)  

    class_head = layers.Dense(units=128, activation='relu')(class_head)
    class_head = layers.Dropout(0.3)(class_head)

    class_output = layers.Dense(units=37, activation="softmax", name="class_output")(class_head)

    # --- BOUNDING BOX HEAD (Shallower & Wider for Regression) ---
    bbox_head = layers.Dense(units=256)(features)
    bbox_head = layers.LeakyReLU(negative_slope=0.1)(bbox_head)

    bbox_head = layers.Dense(units=128)(bbox_head)
    bbox_head = layers.LeakyReLU(negative_slope=0.1)(bbox_head)

    bbox_output = layers.Dense(units=4, activation="sigmoid", name="bbox_output")(bbox_head)

    # --- MODEL COMPILE ---
    logging.info("Compiling model")
    model = keras.Model(inputs=input_layer, outputs=[class_output, bbox_output], name="SOD-modelv0.1")

    model.compile(
    optimizer=Adam(learning_rate=1e-4), # Use a slightly lower LR for transfer learning
    loss={
        "class_output": "sparse_categorical_crossentropy",
        "bbox_output": Huber()
    },
    loss_weights={
        "class_output": 1.0, 
        "bbox_output": 50.0  #  Balanced weight so bbox loss matters to the optimizer
    },
    metrics={
        "class_output": ["accuracy"],
        "bbox_output": ["mae"]
    }
    )

    return model