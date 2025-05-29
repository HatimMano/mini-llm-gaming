# src/__init__.py
"""
Mini-LLM: Architecture ultra-légère pour déploiement massif
"""

__version__ = "0.1.0"
__author__ = "Mini-LLM Team"

from .models.nano_llm import NanoLLM
from .models.micro_llm import MicroLLM
from .models.base_model import BaseLLM

__all__ = ['NanoLLM', 'MicroLLM', 'BaseLLM']