"""PPP: ProtoMotions mimic inference on ovphysx + ovrtx."""

from __future__ import annotations

import os
import sys
from pathlib import Path


__version__ = "0.1.0"


def _ensure_protomotions_on_path() -> None:
    """Make the ProtoMotions source tree importable.

    The conda-installed ``protomotions`` package ships only top-level files
    (the original ``setup.py`` uses ``packages=["protomotions"]``). The full
    source tree (``protomotions.components``, ``protomotions.envs``, …)
    lives in the cloned repo and is normally accessed by running scripts
    from the repo root. We replicate that behaviour by prepending the repo
    root to ``sys.path`` so ``import protomotions.X`` resolves correctly.
    """
    root = os.environ.get("PROTOMOTIONS_REPO")
    candidates = []
    if root:
        candidates.append(Path(root))
    candidates.append(Path("C:/Git/ProtoMotions"))

    for cand in candidates:
        sub = cand / "protomotions" / "components"
        if sub.exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_ensure_protomotions_on_path()


# NOTE: We deliberately do *not* import ``ovrtx`` here. ovrtx and ovphysx
# ship incompatible carb plugin stacks and cannot coexist in the same
# Python process — the one that initializes first wins the carb singletons
# and the other one fails to load its plugins. We sidestep the conflict by
# running ovrtx in a child interpreter (see ppp/remote_renderer.py).
