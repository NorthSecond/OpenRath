"""OpenRath: a torch-like API framework for dynamic multi-agent workflow.

The public surface is currently limited to :mod:`rath.backend`. Higher-level
session and agent abstractions are intentionally not implemented yet.
"""

from rath import backend

__all__ = ["backend"]
