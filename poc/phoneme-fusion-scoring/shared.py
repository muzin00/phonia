"""Frozen inputs and runtime shared with prior, unmodified experiments."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-segment-selection-rl"))
import runtime as rt
from final_fusion import batch, fit
from results import bootstrap, summarize
from verification import Bank, evaluate, load_evaluations

sys.path.insert(0, str(BASE))
__all__ = ["Bank", "batch", "bootstrap", "evaluate", "fit", "summarize"]
ARMS = ("weighted", "weighted_no_kl", "direct")


def configuration(smoke=False):
    config = rt.read(BASE / "config.json")
    if smoke:
        config.update(
            run_directory=config["run_directory"] + "-smoke",
            smoke_test_only=True,
            evaluation_seeds=config["evaluation_seeds"][:1],
            final_fusion_updates=40,
            final_fusion_initial_updates=10,
            final_fusion_validation_every=20,
            bootstrap_replicates=20,
        )
    return config


def source(config):
    return ROOT / config["encoder_run"]


def evaluation_prepared(config):
    return ROOT / config["evaluation_run"] / "prepared"


def encoder_path(config, seed):
    return source(config) / "original" / f"seed-{seed}" / "acoustic/encoder.pt"


def target(run, arm, seed):
    return run / arm / f"seed-{seed}"


def load_encoder(config, seed):
    encoder = rt.create_encoder("statistics_mlp")
    state = rt.torch.load(encoder_path(config, seed), weights_only=True)
    encoder.load_state_dict(state["encoder"])
    encoder.requires_grad_(False).eval()
    return encoder


class CachedEvaluation:
    """Reuse only frozen-encoder embeddings; never cache trainable fusion output."""

    def __init__(self, original, encoder):
        self.original = original
        self.arrays = original.bank.materialize(
            encoder, original.queries, original.claims
        )

    def __getattr__(self, name):
        return getattr(self.original, name)

    @rt.torch.no_grad()
    def scores(self, student):
        student.fusion.eval()
        values = []
        for start in range(0, len(self.labels), 256):
            features = self.bank.features_for(self.arrays, slice(start, start + 256))
            if not features[2][:, :5].all():
                raise ValueError("Frozen evaluation eligibility changed")
            values.append(student.fusion(*features)[0].numpy())
        return rt.np.concatenate(values)


def cached_evaluations(config, encoder, split):
    return {
        c: CachedEvaluation(data, encoder)
        for c, data in load_evaluations(
            config, evaluation_prepared(config), split
        ).items()
    }
