"""Models module for object detection architectures and training routines."""

from . import single_object_detection
from .single_object_detection import sod_modelv01, train_model

__all__ = [
    "single_object_detection",
    "sod_modelv01",
    "train_model",
]