"""Observed-data enrollment reduction, with unchanged reference cutoffs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-live-impostor"))
sys.path.insert(0, str(BASE.parent / "phoneme-live-analysis"))
import diagnose as d
import screen as s

np = s.np


def freeze(config, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    files = {}
    for key in ("screen_run", "prefix_run"):
        source = ROOT / config[key]
        old = s.read(source / "design-freeze.json")
        files.update(old["files"])
        for path in source.rglob("*.json"):
            files[str(path.relative_to(ROOT))] = s.sha(path)
    for path in [
        *BASE.glob("*.py"),
        *BASE.glob("*.json"),
        BASE / "README.md",
        *BASE.glob("tests/*.py"),
    ]:
        files[str(path.relative_to(ROOT))] = s.sha(path)
    for name, digest in files.items():
        assert s.sha(ROOT / name) == digest, name
    s.write(output / "design-freeze.json", {"config": config, "files": files})
    print(f"Frozen {len(files)} files", flush=True)


def read_clips(directory, clips, engine):
    rows, vectors, by_clip = [], [], []
    for clip in clips:
        source = directory / clip["id"]
        segments = list(s.r.s.rows(source / "segments.jsonl"))
        matrix = np.load(source / "embeddings.npy", allow_pickle=False)
        features = np.load(source / "features.npy", allow_pickle=False)
        replay = s.r.s.original.model_embeddings(engine.model, features)
        assert np.array_equal(matrix, replay)
        selected = []
        for row, vector in zip(segments, matrix, strict=True):
            selected.append({**row, "cache_index": len(vectors)})
            vectors.append(vector)
        rows.extend(selected)
        by_clip.append(selected)
    return rows, np.asarray(vectors, dtype=np.float32).reshape(-1, 128), by_clip


def paired(baseline, current, cut):
    key = lambda row: (row["corpus"], row["speaker_id"], row["query_id"])
    assert [key(v) for v in baseline] == [key(v) for v in current]
    delta = []
    transitions = {"accept_to_reject_or_unscored": 0, "reject_or_unscored_to_accept": 0}
    for old, new in zip(baseline, current, strict=True):
        old_accept = old["score"] is not None and old["score"] >= cut
        new_accept = new["score"] is not None and new["score"] >= cut
        transitions["accept_to_reject_or_unscored"] += old_accept and not new_accept
        transitions["reject_or_unscored_to_accept"] += not old_accept and new_accept
        if old["score"] is not None and new["score"] is not None:
            delta.append(new["score"] - old["score"])
    return {
        **transitions,
        "scored_pairs": len(delta),
        "mean_score_difference": float(np.mean(delta)) if delta else None,
        "max_absolute_score_difference": max(map(abs, delta), default=None),
    }


def run(config, output):
    frozen = s.read(output / "design-freeze.json")
    assert frozen["config"] == config
    for name, digest in frozen["files"].items():
        assert s.sha(ROOT / name) == digest, name
    prior = ROOT / config["screen_run"]
    design = s.read(prior / "design-freeze.json")
    source_config = design["config"]
    cuts = s.read(prior / "threshold-freeze.json")["thresholds"][config["arm"]]
    engine = s.r.Runtime(design["live_config"])
    live = ROOT / source_config["live_run"]
    directory = live / "episodes" / design["enrollment_episode_id"]
    episode = s.read(directory / "episode.json")
    _, vectors, by_clip = read_clips(directory, episode["clips"], engine)
    prefix_config = s.read(ROOT / config["prefix_run"] / "design-freeze.json")["config"]
    profiles, meta = {}, {}
    for n in config["prefixes"]:
        rows = [row for group in by_clip[:n] for row in group]
        profile, selected, eligible = d.make_profile(rows, vectors, prefix_config)
        profiles[n] = profile
        meta[n] = {
            "eligible": eligible,
            "recording_seconds": sum(c["seconds"] for c in episode["clips"][:n]),
            "selected_vowel_counts": {v: len(selected.get(v, [])) for v in s.VOWELS},
            "phones": sorted(profile),
        }
    saved = s.load_profile(source_config, design, engine)
    assert saved.keys() == profiles[20].keys()
    assert all(np.array_equal(saved[p], profiles[20][p]) for p in saved)
    common = sorted(
        set.intersection(*(set(p) for n, p in profiles.items() if meta[n]["eligible"]))
    )
    genuine_inputs = []
    for identifier in design["live_query_ids"]:
        directory = live / "episodes" / identifier
        query = s.read(directory / "episode.json")
        rows, matrix, _ = read_clips(directory, query["clips"], engine)
        groups = {}
        for row in rows:
            groups.setdefault(row["phoneme"], []).append(row["cache_index"])
        genuine_inputs.append(({**query, "groups": groups}, matrix))
    previous_prefix = s.read(ROOT / config["prefix_run"] / "results.json")
    all_results, checks = {}, {}
    for mode in config["modes"]:
        all_results[mode] = {}
        for n, profile in profiles.items():
            if not meta[n]["eligible"]:
                continue
            active = profile if mode == "native" else {p: profile[p] for p in common}
            genuine = []
            for q, matrix in genuine_inputs:
                value = d.score(active, True, q, matrix, active)
                genuine.append(
                    {
                        **value,
                        "corpus": "live",
                        "speaker_id": q["participant"],
                        "clips": len(q["clips"]),
                    }
                )
            expected = previous_prefix["prefixes"][str(n)][mode]
            assert [v["query_id"] for v in genuine] == [v["query_id"] for v in expected]
            assert [v["score"] for v in genuine] == [v["score"] for v in expected]
            impostors = []
            for corpus in source_config["corpora"]:
                scores, audit = s.corpus_scores(
                    {**source_config, "arms": [config["arm"]]},
                    design,
                    corpus,
                    "test",
                    active,
                    engine,
                )
                impostors.extend(scores[config["arm"]])
                checks[f"{mode}/{n}/{corpus}"] = audit
            if mode == "native" and n == 20:
                old = [
                    row
                    for c in source_config["corpora"]
                    for row in s.read(prior / "test" / f"{c}-scores.json")[
                        config["arm"]
                    ]
                ]
                assert impostors == old
            all_results[mode][n] = {"genuine": genuine, "impostors": impostors}
            s.write(output / mode / f"prefix-{n}.json", all_results[mode][n])
            print(f"Scored {mode}/{n}", flush=True)
    summary = {}
    for mode, prefixes in all_results.items():
        summary[mode] = {}
        for n, values in prefixes.items():
            genuine, impostors = values["genuine"], values["impostors"]
            g = [v["score"] for v in genuine if v["score"] is not None]
            i = [v["score"] for v in impostors if v["score"] is not None]
            summary[mode][n] = {
                "genuine": s.describe(genuine),
                "impostors": s.describe(impostors),
                "observed_gap": min(g) - max(i),
                "points": {
                    target: {
                        "threshold": cut,
                        "genuine": s.rates(genuine, cut),
                        "genuine_with_audio": s.rates(
                            [v for v in genuine if v["clips"]], cut
                        ),
                        "impostors": s.rates(impostors, cut),
                        "paired_genuine": paired(prefixes[20]["genuine"], genuine, cut),
                        "paired_impostors": paired(
                            prefixes[20]["impostors"], impostors, cut
                        ),
                    }
                    for target, cut in cuts.items()
                },
            }
    s.write(
        output / "results.json",
        {
            "config": config,
            "enrollment": meta,
            "common_phones": common,
            "results": summary,
            "checks": checks,
            "original_profile_and_scores_replayed": True,
            "live_protocol_changed": False,
        },
    )
    print(json_dump(summary), flush=True)


def json_dump(value):
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    config = s.read(BASE / "config.json")
    output = ROOT / config["run_directory"]
    if args.phase == "freeze":
        freeze(config, output)
    else:
        run(config, output)


if __name__ == "__main__":
    main()
