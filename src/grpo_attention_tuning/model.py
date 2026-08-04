"""Thin re-export of the unified LatentModel from src/latent/layers.py.

The stage-2 model used to be a near-duplicate of the stage-1 one (only the
num_logits_to_keep/logits_to_keep kwarg differed); the unified LatentModel
accepts both spellings, so both stages share a single implementation.
"""
import os
import sys

_latent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "latent")
if _latent_dir not in sys.path:
    sys.path.insert(0, _latent_dir)

from layers import LatentModel, SelfAttentionLayer, VanillaRoPE  # noqa: F401,E402
