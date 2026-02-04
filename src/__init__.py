"""
=============================================================================
Sign Language Recognition - Core ML Modules
=============================================================================
File: src/__init__.py
Description: Package initializer for the core ML modules. Exports the main
             classes and functions for preprocessing, model building, and
             inference.
=============================================================================
"""

from .preprocessing import MediaPipeExtractor, preprocess_dataset
from .model import SignLanguageClassifier, create_lstm_model, create_transformer_model
from .inference import SignLanguagePredictor, benchmark_inference

__all__ = [
    'MediaPipeExtractor',
    'preprocess_dataset',
    'SignLanguageClassifier',
    'create_lstm_model',
    'create_transformer_model',
    'SignLanguagePredictor',
    'benchmark_inference'
]