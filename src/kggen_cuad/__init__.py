"""
KGGEN-CUAD: Knowledge Graph Generator for Legal Contracts

This package implements the KGGen methodology for extracting structured
knowledge graphs from the CUAD (Contract Understanding Atticus Dataset)
for legal contract analysis.
"""

__version__ = "0.1.0"
__author__ = "KGGEN Team"

from kggen_cuad.config import get_settings

__all__ = ["get_settings", "__version__"]
