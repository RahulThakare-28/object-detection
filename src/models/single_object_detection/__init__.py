"""Single Object Detection (SOD) module."""

from .sod_model import sod_modelv01
from .sod_training import train_model

__all__ = [
    "sod_modelv01",
    "train_model",
]
