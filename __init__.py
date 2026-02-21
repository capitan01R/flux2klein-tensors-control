"""
FLUX.2 Klein Tensor Debiaser — ComfyUI Custom Node
===================================================
Full 201 tensor control for FLUX.2 Klein 9B.

Installation:
  Place this folder in ComfyUI/custom_nodes/

Files:
  __init__.py                         - This file (node registration)
  flux_klein_tensor_debiaser_node.py  - Node implementation
  flux_klein_tensor_debiaser_ui.js    - UI extension (goes in js/ subfolder or web/extensions/)
"""

from .flux_klein_tensor_debiaser_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
