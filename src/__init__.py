"""
=============================================================================
Sign Language Recognition - Core ML Modules
=============================================================================
File: src/__init__.py
Description: Package initializer for the core ML modules.
=============================================================================
"""

# Only import what actually exists
try:
    from .preprocessing import MediaPipeExtractor
except ImportError:
    pass

__all__ = []

