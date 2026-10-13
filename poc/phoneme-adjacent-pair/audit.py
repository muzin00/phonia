"""Check training parity, exact PCM support and saved-score arithmetic."""

import hashlib
import wave

from pair_common import (
    ARMS,
    ROOT,
    check,
    features,
    metrics,
    model_hash,
    np,
    pipeline,
    pooled,
    read,
    sha,
    sources,
    torch,
    write,
)


def independent_eer(values, labels):
    if not labels.any() or labels.all():
        return None
    _, inverse = np.unique(values, return_inverse=True)
    positives = np.bincount(inverse, weights=labels)
    negatives = np.bincount(inverse, weights=~labels, minlength=len(positives))
    far = np.r_[1.0, 1.0 - np.cumsum(negatives) / negatives.sum()]
    frr = np.r_[0.0, np.cumsum(positives) / positives.sum()]
    d = far - frr
    exact = np.flatnonzero(d == 0)
    if len(exact):
        return float(far[exact[0]])
    right = int(np.flatnonzero(d < 0)[0])
    left = right - 1
    fraction = d[left] / (d[left] - d[right])
    return float(far[left] + fraction * (far[right] - far[left]))


def audit(config, run):
    for name in (
        "design-freeze.json",
        "models-freeze.json",
        "thresholds-freeze.json",
        "test-prepared.json",
    ):
        check(read(run / name)["files"])
    selection = read(run / "pair-selection.json")
    thresholds = read(run / "thresholds-freeze.json")
    assert thresholds["models_freeze_sha256"] == sha(run / "models-freeze.json")
    pairs = selection["selected"]
    eligible = []
    for p, v in selection["support"].items():
        valid = all(
            t["speakers_with_two"] / t["total_speakers"]
            >= config["minimum_train_speaker_fraction_per_corpus"]
            for t in v["train"].values()
        ) and all(
            t["enrollment_speakers"] == t["total_speakers"]
            and t["query_count"] / t["query_total"]
            >= config["minimum_validation_query_fraction"]
            for t in v["validation"].values()
        )
        assert valid == v["eligible"]
        if valid:
            eligible.append(p)
    expected = sorted(
        eligible, key=lambda p: (-selection["support"][p]["minimum_query_fraction"], p)
    )[: config["maximum_pair_types"]]
    assert sorted(expected) == pairs
    assert not selection["test_read_for_selection"]
    training = read(run / "data/train-train/units.json")
    speakers = sorted({u["speaker_id"] for u in training})
    train_hashes = {u["source_sha256"] for u in training}
    seen_speakers = {"train": set(speakers), "validation": set(), "test": set()}
    pipe, _ = pipeline(config)
    feature_error, total_units, total_frames = 0.0, 0, 0
    for split in ("train", "validation", "test"):
        for c, spec in sources(config, split).items():
            directory = run / "data" / f"{c}-{split}"
            units = read(directory / "units.json")
            cache = np.load(
                directory / "features.npy", mmap_mode="r", allow_pickle=False
            )
            original = features(spec)
            assert cache.shape == (len(units), 3, 128) and np.isfinite(cache).all()
            assert len({u["unit_id"] for u in units}) == len(units)
            frames = 0
            for u in units:
                a, b, end = u["frames"]
                assert a >= 0 and min(b - a, end - b) >= config["minimum_phone_frames"]
                assert (
                    end - a <= config["maximum_pair_frames"]
                    and (b - a) + (end - b) == end - a
                )
                assert u["pair"] in pairs
                frames += end - a
            for start in range(0, len(units), 4096):
                indices = np.asarray(
                    [u["indices"] for u in units[start : start + 4096]]
                )
                np.testing.assert_array_equal(
                    cache[start : start + 4096, :2], original[indices]
                )
            probes = np.random.default_rng(config["selection_seed"]).choice(
                len(units), size=min(100, len(units)), replace=False
            )
            for i in probes:
                u = units[i]
                a, b, end = u["frames"]
                with wave.open(str(ROOT / u["source_file"])) as wav:
                    wav.setpos(a)
                    pcm = (
                        np.frombuffer(wav.readframes(end - a), dtype="<i2").astype(
                            np.float32
                        )
                        / 32768
                    )
                assert len(pcm) == end - a
                direct = np.stack(
                    [
                        pooled(pipe, pcm[: b - a], u["unit_id"]),
                        pooled(pipe, pcm[b - a :], u["unit_id"]),
                        pooled(pipe, pcm, u["unit_id"]),
                    ]
                )
                error = float(np.max(np.abs(direct - cache[i])))
                assert error < 2e-6
                feature_error = max(feature_error, error)
            receipt = read(directory / "complete.json")
            assert (
                receipt["unique_frames_split"]
                == receipt["unique_frames_joined"]
                == frames
            )
            total_units += len(units)
            total_frames += frames
            if split != "train":
                data = read(directory / "inputs.json")
                seen_speakers[split].update(data["speakers"])
                assert not {u["source_sha256"] for u in units} & train_hashes
                enrolled = {
                    units[i]["source_sha256"]
                    for profile in data["profiles"].values()
                    for ids in profile.values()
                    for i in ids
                }
                queried = {
                    units[i]["source_sha256"]
                    for q in data["queries"]
                    for ids in q["groups"].values()
                    for i in ids
                }
                assert not enrolled & queried
    assert not seen_speakers["train"] & (
        seen_speakers["validation"] | seen_speakers["test"]
    )
    assert not seen_speakers["validation"] & seen_speakers["test"]
    updates = config["updates_per_pair_type"] * len(pairs)
    training_checks = {}
    pair_ids = np.array([pairs.index(u["pair"]) for u in training])
    owner_ids = np.array([speakers.index(u["speaker_id"]) for u in training])
    for seed in config["seeds"]:
        summaries = [read(run / f"seed-{seed}/{arm}/summary.json") for arm in ARMS]
        assert summaries[0]["initial"] == summaries[1]["initial"]
        assert summaries[0]["identity"] == summaries[1]["identity"]
        with np.load(run / f"seed-{seed}/schedule.npz", allow_pickle=False) as saved:
            ids, labels = saved["ids"], saved["labels"]
            assert ids.shape == labels.shape == (updates, 100)
            assert np.all(ids.reshape(-1, 20)[:, ::2] != ids.reshape(-1, 20)[:, 1::2])
            np.testing.assert_array_equal(owner_ids[ids], labels)
            np.testing.assert_array_equal(
                pair_ids[ids].reshape(-1, 20),
                np.broadcast_to(
                    (np.arange(updates * 5) % len(pairs))[:, None], (updates * 5, 20)
                ),
            )
            groups = labels.reshape(-1, 20)
            np.testing.assert_array_equal(groups[:, ::2], groups[:, 1::2])
            assert np.all(np.diff(np.sort(groups[:, ::2], axis=1), axis=1) > 0)
            digest = hashlib.sha256(ids.tobytes() + labels.tobytes()).hexdigest()
            assert digest == summaries[0]["identity"]["schedule_sha256"]
        for arm, summary in zip(ARMS, summaries, strict=True):
            directory = run / f"seed-{seed}/{arm}"
            check(read(directory / "complete.json")["files"])
            saved = torch.load(
                directory / "last.pt", weights_only=False, map_location="cpu"
            )
            exported = torch.load(
                directory / "encoder.pt", weights_only=True, map_location="cpu"
            )
            assert (
                saved["update"] == exported["update"] == summary["updates"] == updates
            )
            assert (
                model_hash(saved["model"])
                == model_hash(exported["model"])
                == summary["final_encoder_sha256"]
            )
            assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {
                updates
            }
            assert summary["encoder_parameters"] == 65920
        training_checks[str(seed)] = {
            "same_initialization": True,
            "same_schedule": True,
            "updates": updates,
        }
    maximum_metric_error, score_count = 0.0, 0
    for split in ("validation", "test"):
        records = read(run / f"{split}-metrics.json")
        pair_records = read(run / f"{split}-pairs.json")
        for c in config["corpora"]:
            for seed in config["seeds"]:
                with np.load(
                    run / "scores" / f"{c}-{seed}-{split}.npz", allow_pickle=False
                ) as scores:
                    labels = scores["labels"]
                    np.testing.assert_array_equal(
                        labels, scores["query_speakers"] == scores["claims"]
                    )
                    np.testing.assert_array_equal(
                        np.isfinite(scores["split"]), np.isfinite(scores["joined"])
                    )
                    n = int(scores["claims"].max()) + 1
                    for arm in ARMS:
                        for row in records:
                            if (
                                row["corpus"] != c
                                or row["seed"] != seed
                                or row["arm"] != arm
                            ):
                                continue
                            role = scores["roles"] == row["role"]
                            keep = role & np.isfinite(scores[arm])
                            x, y = scores[arm][keep], labels[keep]
                            t = row["threshold"]
                            assert t == thresholds["thresholds"][f"{c}/{seed}/{arm}"]
                            if split == "validation" and row["role"] == "verification":
                                expected_threshold, _ = metrics.far_target_threshold(
                                    x, y, 0.01
                                )
                                assert metrics.json_threshold(expected_threshold) == t
                            threshold = (
                                np.inf if t == "+inf" else -np.inf if t == "-inf" else t
                            )
                            accepted = x >= threshold
                            positive = int(labels[role].sum())
                            negative = int((~labels[role]).sum())
                            expected = {
                                "eer": independent_eer(x, y),
                                "coverage": int(y.sum()) / positive,
                                "far": int(np.sum(accepted & ~y)) / negative,
                                "frr": (positive - int(np.sum(accepted & y)))
                                / positive,
                            }
                            for k, v in expected.items():
                                difference = abs(v - row[k])
                                assert difference < 1e-12
                                maximum_metric_error = max(
                                    maximum_metric_error, difference
                                )
                            score_count += len(x)
                        for p, pair in enumerate(pairs):
                            mask = (
                                scores["roles"].reshape(-1, n)[:, 0] == "verification"
                            ) & scores["mask"][:, p]
                            x = scores[f"{arm}_components"][mask, :, p].ravel()
                            y = labels.reshape(-1, n)[mask].ravel()
                            expected = independent_eer(x, y)
                            row = next(
                                r
                                for r in pair_records
                                if r["corpus"] == c
                                and r["seed"] == seed
                                and r["arm"] == arm
                                and r["pair"] == pair
                            )
                            if expected is None:
                                assert row["eer"] is None
                            else:
                                assert abs(expected - row["eer"]) < 1e-12
    with np.load(run / "bootstrap.npz", allow_pickle=False) as distributions:
        for c, interval in read(run / "bootstrap.json")["eer_delta_ci95_pp"].items():
            np.testing.assert_array_equal(
                100 * np.quantile(distributions[c], [0.025, 0.975]), interval
            )
    assert (run / "thresholds-freeze.json").stat().st_mtime <= (
        run / "test-prepared.json"
    ).stat().st_mtime
    result = {
        "source_hashes_verified": True,
        "pair_selection_reconstructed": True,
        "same_pcm_frames": True,
        "units_checked": total_units,
        "pcm_frames_checked": total_frames,
        "all_separate_features_exactly_match_original": True,
        "maximum_feature_difference": feature_error,
        "training": training_checks,
        "saved_score_values_checked": score_count,
        "maximum_metric_difference": maximum_metric_error,
        "no_speaker_or_wav_hash_overlap": True,
        "test_after_validation_threshold_freeze": True,
    }
    write(run / "independent-audit.json", result, immutable=True)
    return result
