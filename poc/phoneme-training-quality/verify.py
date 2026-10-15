"""Load the screened view with existing learners, without training or EER selection."""

import sys
from pathlib import Path

from prepare import BASE, ROOT, checked, pin, read, write


def main():
    config = read(BASE / "config.json")
    run = ROOT / config["run_directory"]
    seal = read(run / "complete.json")
    checked(seal["inputs"])
    checked(seal["outputs"])
    sys.path.insert(0, str(BASE.parent / "phoneme-segment-selection-rl"))
    import runtime as rt
    from pool import Candidates, Pool
    from verification import Bank

    rt.torch.set_num_threads(1)
    directory = run / "screened"
    pool = Pool(directory, directory)
    ids = rt.np.load(directory / "source-segment-indices.npy", allow_pickle=False)
    with rt.np.load(run / "decisions.npz", allow_pickle=False) as decisions:
        if not rt.np.all(decisions["status"][ids] == 0):
            raise ValueError("Non-accepted interval leaked into training")
    sampler = Candidates(pool, config["review_seed"], count=2)
    for step in range(8):
        batch = sampler.draw(step)
        pairs = batch["pairs"][batch["mask"]]
        if rt.np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("Same interval reused as its own positive")
        for field in (pool.segment_speaker, pool.segment_phone):
            if not rt.np.array_equal(field[pairs[:, 0]], field[pairs[:, 1]]):
                raise ValueError("Positive pair ownership mismatch")
    info = read(directory / "train-inputs.json")
    bank = Bank(
        info, pool.features, rt.np.load(directory / "train-metadata.npy"), pool.phones
    )
    rt.torch.manual_seed(config["review_seed"])
    encoder = rt.create_encoder("statistics_mlp")
    # Exercise a training query from every speaker with a fresh encoder; no optimizer.
    queries = rt.np.array(
        [rt.np.flatnonzero(bank.owners == s)[0] for s in range(len(bank.speakers))]
    )
    for start in range(0, len(queries), 32):
        q = queries[start : start + 32]
        values = bank.materialize(encoder, q, bank.owners[q])
        features, components, mask = bank.features_for(values, slice(None))
        if (
            not mask[:, :5].all()
            or not rt.torch.isfinite(features).all()
            or not rt.torch.isfinite(components).all()
        ):
            raise ValueError("Invalid Transformer training input")
    report = {
        "status": "passed",
        "acoustic_segments": len(pool.features),
        "acoustic_speakers": len(pool.speakers),
        "acoustic_phones": len(pool.phones),
        "fusion_speakers_checked": len(bank.speakers),
        "fusion_queries": len(bank.owners),
        "held_or_rejected_training_indices": 0,
        "optimizer_updates": 0,
        "evaluation_metrics_computed": False,
        "inputs": pin(
            [
                run / "complete.json",
                Path(__file__).resolve(),
                BASE.parent / "phoneme-segment-selection-rl/pool.py",
                BASE.parent / "phoneme-segment-selection-rl/verification.py",
            ]
        ),
    }
    write(run / "compatibility-check.json", report)
    print({k: v for k, v in report.items() if k != "inputs"})


if __name__ == "__main__":
    main()
