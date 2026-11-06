"""Reconstruction: capture video -> frames -> masks -> camera poses -> dense cloud -> mesh -> GLB.

Each stage is a module of plain functions; `run.py` strings them together for one capture.
"""


class PipelineError(Exception):
    """A failure the capturer or organiser can act on. The message is shown to them as is."""
