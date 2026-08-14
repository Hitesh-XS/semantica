"""Semantica integration for Google's Agent Development Kit (ADK)."""

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from .decision_tools import semantica_decision_tools
from .kg_tools import semantica_kg_tools
from .session_service import SemanticaSessionService

__all__ = [
    "ADK_AVAILABLE",
    "SemanticaSessionService",
    "semantica_decision_tools",
    "semantica_kg_tools",
]

__version__ = "0.1.0"
