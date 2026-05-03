"""Arabic Restaurant Complaints Classifier — application package.

Public exports:
    EnsembleClassifier  — load + predict with the 4-model 8-class ensemble
    clean_arabic         — Lana's text cleaning pipeline
"""
from app.ensemble_inference import EnsembleClassifier, clean_arabic

__version__ = "0.5.0"
__all__ = ["EnsembleClassifier", "clean_arabic", "__version__"]
