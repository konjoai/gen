"""gen (源 — source / origin): Konjo architecture-research scaffold.

A reusable synthetic gauntlet, the `SequenceMixer` plug interface, reference
baselines, the 30-run statistical gate, and the four-gate decision ledger. The
scaffold discriminates a capable mixer from an incapable one in minutes; it
implements no novel primitive. See docs/CHARTER.md.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Reproducibility: a single place to fix all RNG + deterministic flags.
import os
import platform
import random

import numpy as np
import torch


def set_determinism(seed: int = 0) -> None:
    """Fix seeds and enable deterministic algorithms. Call before any run."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_header(seed: int | None = None) -> dict[str, str]:
    """Version/platform provenance for every report header (charter §2 KCQF-lite)."""
    header = {
        "gen_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if seed is not None:
        header["seed"] = str(seed)
    return header


__all__ = ["__version__", "env_header", "set_determinism"]
