"""Typed errors raised at the model boundary."""

from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """The model server cannot be reached or returned no usable answer."""
