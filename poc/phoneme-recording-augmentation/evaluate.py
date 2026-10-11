"""Calibrate all final models before new test inference on observed cohorts."""

import argparse

import train as e

s, np, t, torch = e.s, e.np, e.t, e.torch


def load(config, run, name):
    old = s.read_json(run / "design-freeze.json")["source_config"]
    if name == "original":
        return s.original.load_model(
            e.ROOT / old["encoder_run"] / "trials" / old["encoder_trial"]
        )[0]
    model = e.learner.create_encoder("statistics_mlp")
    value = torch.load(run / name / "encoder.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(value["model"])
    return model.eval()


def names(config):
    return [
        "original",
        *[f"{arm}-{seed}" for arm in config["arms"] for seed in config["seeds"]],
    ]


def evaluate(config, run, split):
    design = e.verify(config, run)
    old = design["source_config"]
    source = e.ROOT / config["quality_run"]
    if split == "test":
        frozen = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", frozen["design_sha256"])
        for name, digest in frozen["files"].items():
            s.checked(e.ROOT / name, digest)
    selected = {
        r["query"]["query_id"]
        for r in s.read_json(source / "selection.json")["stress"][split]
    }
    jvs = t.jvs_data(old, split)
    jvs = {**jvs, "queries": [q for q in jvs["queries"] if q["query_id"] in selected]}
    cv = s.read_json(e.ROOT / old["cv_run"] / split / "inputs.json")
    jvs_features = np.memmap(
        e.ROOT / old["encoder_run"] / f"{split}-features.f32", mode="r", dtype="<f4"
    ).reshape(-1, 128)
    cv_features = np.load(
        e.ROOT / old["cv_run"] / split / "features.npy", allow_pickle=False
    )
    points = {}
    for name in names(config):
        model = load(config, run, name)
        dest = run / split / name
        dest.mkdir(parents=True, exist_ok=True)
        vectors = s.original.model_embeddings(model, jvs_features)
        np.save(dest / "JVS-embeddings.npy", vectors, allow_pickle=False)
        registration = t.profiles(jvs, vectors)
        for condition in config["conditions"]:
            values = []
            probe_vectors = {}
            for path in sorted((source / split / condition).glob("*/prepared.json")):
                prepared = s.read_json(path)
                with np.load(path.parent / "features.npz", allow_pickle=False) as z:
                    features = z["fixed"]
                embeddings = s.original.model_embeddings(model, features)
                probe_vectors[prepared["query"]["query_id"]] = embeddings
                values.extend(
                    t.score(
                        prepared["query"],
                        jvs,
                        embeddings,
                        prepared["fixed"],
                        registration,
                        "fixed",
                        split,
                    )
                )
            s.write_rows(dest / f"{condition}-scores.jsonl", values)
            np.savez(dest / f"{condition}-query-embeddings.npz", **probe_vectors)
            if split == "validation" and condition == "clean":
                points[f"{name}/JVS"] = {
                    p["name"]: s.original.thresholds(s.apply_policy(values, p))
                    for p in old["policies"]
                }
        vectors = s.original.model_embeddings(model, cv_features)
        np.save(dest / "CV-embeddings.npy", vectors, allow_pickle=False)
        values = t.h.candidates(cv, vectors)
        s.write_rows(dest / "CV-scores.jsonl", values)
        if split == "validation":
            points[f"{name}/CV"] = {
                p["name"]: s.original.thresholds(s.apply_policy(values, p))
                for p in old["policies"]
            }
        print(f"Completed {split} inference {name}", flush=True)
    if split == "validation":
        files = {
            t.rel(p): s.sha256_file(p) for p in (run / split).rglob("*") if p.is_file()
        }
        for name in names(config):
            if name != "original":
                t.pin(files, run / name / "encoder.pt")
                t.pin(files, run / name / "complete.json")
        s.write_json(
            run / "threshold-freeze.json",
            {
                "thresholds": points,
                "files": files,
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "status": "all_models_and_calibration_frozen_before_test_inference",
            },
        )
    e.verify(config, run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("validation", "test"))
    args = parser.parse_args()
    config = s.read_json(e.CONFIG)
    torch.set_num_threads(1)
    evaluate(config, e.ROOT / config["run_directory"], args.split)


if __name__ == "__main__":
    main()
