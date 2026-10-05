"""Phase 6 orchestration; inference and scoring remain in Phases 4 and 5."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
for name in (
    "phoneme-speaker-encoder",
    "phoneme-user-registration",
    "phoneme-verification",
):
    sys.path.insert(0, str(BASE.parent / name))
sys.path.insert(0, str(BASE.parent / "phoneme-speaker-encoder/scripts"))
