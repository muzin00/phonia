"""Independently verify checkpoint selection, paired starts and published metrics."""

import hashlib

from common import (
    ARMS,
    VARIANTS,
    assert_nested_equal,
    check,
    evaluate_eers,
    evaluation_data,
    evaluation_inputs,
    initial_model,
    model_hash,
    np,
    pin,
    positions,
    prior,
    read,
    sha,
    torch,
    training_data,
    write,
)
from evaluation import numeric_threshold
from selection import Teacher, choose, select_indices, teacher_loss
from trainer import optimizer_for, tree_hash


def audit(config, run):
    check(read(run / "design-freeze.json")["files"])
    check(read(run / "models-freeze.json")["files"])
    selection = read(run / "selection-freeze.json")
    check(selection["selection_files"])
    assert selection["models_sha256"] == sha(run / "models-freeze.json")
    training = {}
    for seed in config["encoder_seeds"]:
        initial = initial_model(config, seed)
        arrays, owners, pools = training_data(config, seed)
        with np.load(run / f"draws-{seed}.npz", allow_pickle=False) as archive:
            draws = {k: archive[k] for k in archive.files}
        actions = read(run / "design-freeze.json")["actions"][str(seed)]
        histories = []
        for arm in ARMS:
            directory = run / f"acoustic-{seed}/{arm}"
            check(read(directory / "complete.json")["files"])
            saved = torch.load(
                directory / "last.pt", weights_only=False, map_location="cpu"
            )
            history = read(directory / "history.json")
            assert saved["history"] == history
            assert saved["blocks"] == read(directory / "blocks.json")
            assert saved["initial_model_sha256"] == model_hash(initial.state_dict())
            assert saved["design_sha256"] == sha(run / "design-freeze.json")
            assert saved["update"] == config["updates"]
            assert [h["update"] for h in history] == list(
                range(0, config["updates"] + 1, config["validation_every"])
            )
            histories.append(history[0])
            best = history[0]
            for row in history:
                assert row["macro_eer"] == sum(row["selection_eer"].values()) / len(
                    config["corpora"]
                )
                if row["macro_eer"] < best["macro_eer"] - 1e-12:
                    best = row
            assert saved["best_bundle"]["update"] == best["update"]
            assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {
                config["updates"]
            }
            assert {v["lr"] for v in saved["optimizer"]["param_groups"]} == {
                config["learning_rate"]
            }
            for mode in ("selected", "final"):
                bundle = torch.load(directory / f"{mode}.pt", weights_only=True)
                expected = (
                    saved["best_bundle"]["state_dict"]
                    if mode == "selected"
                    else saved["model"]
                )
                assert_nested_equal(bundle["state_dict"], expected)
                assert bundle["update"] == (
                    best["update"] if mode == "selected" else config["updates"]
                )
            teacher = Teacher(len(actions), config["teacher_exploration"])
            teacher_optimizer = torch.optim.Adam(
                teacher.parameters(), lr=config["teacher_learning_rate"]
            )
            parent_model = model_hash(initial.state_dict())
            parent_optimizer = tree_hash(optimizer_for(config, initial).state_dict())
            current_reward = (
                evaluate_eers(initial, evaluation_data(config, run, seed, "reward"))
                if arm == "rl"
                else None
            )
            rewards, committed_rewards = [], []
            assert len(saved["blocks"]) == config["updates"] // config["block_updates"]
            assert tuple(saved["selected_indices"].shape) == (
                config["updates"],
                config["queries_per_update"],
            )
            for number, block in enumerate(saved["blocks"]):
                start = number * config["block_updates"]
                assert block["start"] == start
                assert block["parent_model_sha256"] == parent_model
                assert block["parent_optimizer_sha256"] == parent_optimizer
                if arm == "rl":
                    context = [
                        1.0,
                        start / config["updates"],
                        sum(current_reward.values()) / len(current_reward),
                    ]
                    assert block["context"] == context
                    probabilities = teacher.probabilities(context)
                    assert probabilities.detach().tolist() == block["probabilities"]
                    assert block["teacher_before"] == teacher.theta.detach().tolist()
                    choices = [
                        choose(probabilities, draw) for draw in draws["actions"][number]
                    ]
                    assert block["choices"] == choices
                    assert block["draws"] == draws["actions"][number].tolist()
                    phones = [actions[i] for i in choices]
                    assert block["phones"] == phones
                    assert (
                        block["committed_rollout"] == 0
                        and block["committed_phone"] == phones[0]
                    )
                    assert set(block["branches"]) == {str(p) for p in [-1, *phones]}
                    control = block["branches"]["-1"]["reward_eer"]
                    expected_rewards = [
                        (
                            sum(control.values())
                            - sum(block["branches"][str(p)]["reward_eer"].values())
                        )
                        / len(control)
                        for p in phones
                    ]
                    np.testing.assert_allclose(
                        expected_rewards, block["rewards"], rtol=0, atol=1e-15
                    )
                    teacher_optimizer.zero_grad(set_to_none=True)
                    loss = teacher_loss(
                        probabilities,
                        choices,
                        block["rewards"],
                        config["teacher_uniform_kl"],
                    )
                    assert float(loss.detach()) == block["teacher_loss"]
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        teacher.parameters(),
                        config["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    assert float(norm) == block["teacher_gradient_norm"]
                    teacher_optimizer.step()
                    assert teacher.theta.detach().tolist() == block["teacher_after"]
                    rewards.extend(block["rewards"])
                    committed_rewards.append(block["rewards"][0])
                    current_reward = block["branches"][str(phones[0])]["reward_eer"]
                else:
                    phone = (
                        -1
                        if arm == "uniform"
                        else actions[
                            choose(
                                torch.ones(len(actions), dtype=torch.float64)
                                / len(actions),
                                draws["actions"][number, 0],
                            )
                        ]
                    )
                    assert block["committed_phone"] == phone
                    assert set(block["branches"]) == {str(phone)}
                assert (
                    block["actual_student_updates"]
                    == len(block["branches"]) * config["block_updates"]
                )
                for phone, branch in block["branches"].items():
                    chosen = np.array(
                        [
                            select_indices(config, pools, draws["keys"], u, int(phone))
                            for u in range(start, start + config["block_updates"])
                        ]
                    )
                    assert (
                        hashlib.sha256(chosen.tobytes()).hexdigest()
                        == branch["indices_sha256"]
                    )
                    available = pools["common"][start : start + len(chosen)]
                    counts = (
                        np.take_along_axis(available, chosen[..., None], axis=1)
                        .sum(axis=(0, 1))
                        .tolist()
                    )
                    assert branch["phone_pair_counts"] == counts
                    if int(phone) == block["committed_phone"]:
                        np.testing.assert_array_equal(
                            chosen,
                            saved["selected_indices"][
                                start : start + len(chosen)
                            ].numpy(),
                        )
                committed = block["branches"][str(block["committed_phone"])]
                parent_model, parent_optimizer = (
                    committed["model_sha256"],
                    committed["optimizer_sha256"],
                )
            assert parent_model == model_hash(saved["model"])
            assert parent_optimizer == tree_hash(saved["optimizer"])
            assert_nested_equal(teacher.state_dict(), saved["teacher"])
            assert_nested_equal(
                teacher_optimizer.state_dict(), saved["teacher_optimizer"]
            )
            if arm == "rl":
                last = initial_model(config, seed)
                last.load_state_dict(saved["model"])
                assert (
                    evaluate_eers(last, evaluation_data(config, run, seed, "reward"))
                    == saved["current_reward"]
                    == current_reward
                )
            training[f"{seed}/{arm}"] = {
                "selected_update": best["update"],
                "selected_selection_macro_eer": best["macro_eer"],
                "actual_student_updates": sum(
                    b["actual_student_updates"] for b in saved["blocks"]
                ),
                "positive_rollout_rewards": sum(r > 1e-12 for r in rewards),
                "negative_rollout_rewards": sum(r < -1e-12 for r in rewards),
                "negative_reward_branches_committed": sum(
                    r < -1e-12 for r in committed_rewards
                ),
                "mean_reward_pp": 100 * float(np.mean(rewards)) if rewards else None,
            }
        assert all(h == histories[0] for h in histories)
        del arrays, owners, pools
    records = read(run / "metrics.json")
    max_error, count = 0.0, 0
    for seed in config["encoder_seeds"]:
        for split in ("selection", "test"):
            for corpus in config["corpora"]:
                arrays, data, rows = evaluation_inputs(config, run, seed, corpus, split)
                del arrays
                lookup = {r["_source_index"]: r for r in rows}
                with np.load(
                    run / "scores" / f"{corpus}-{seed}-{split}.npz", allow_pickle=False
                ) as scores:
                    ids, labels = scores["ids"], scores["labels"]
                    np.testing.assert_array_equal(ids, positions(rows))
                    np.testing.assert_array_equal(
                        labels, [lookup[i]["is_genuine"] for i in ids]
                    )
                    np.testing.assert_array_equal(
                        scores["roles"], [lookup[i]["role"] for i in ids]
                    )
                    np.testing.assert_array_equal(
                        scores["claims"], ids % len(data["speakers"])
                    )
                    np.testing.assert_array_equal(
                        scores["query_speakers"],
                        [data["speakers"].index(lookup[i]["speaker_id"]) for i in ids],
                    )
                    for variant in VARIANTS:
                        values = scores[variant]
                        point = selection["thresholds"][f"{corpus}/{seed}/{variant}"]
                        numeric = numeric_threshold(point)
                        if split == "selection":
                            keep = scores["roles"] == "verification"
                            expected, _ = prior.metrics.far_target_threshold(
                                values[keep], labels[keep], 0.01
                            )
                            assert prior.metrics.json_threshold(expected) == point
                        for role in sorted(set(scores["roles"])):
                            keep = scores["roles"] == role
                            x, y = values[keep], labels[keep]
                            original = [r for r in rows if r["role"] == role]
                            positive = sum(r["is_genuine"] for r in original)
                            negative = len(original) - positive
                            accepted = x >= numeric
                            expected = {
                                "eer": prior.independent_eer(x, y),
                                "far": int(np.sum(accepted & ~y)) / negative,
                                "frr": (positive - int(np.sum(accepted & y)))
                                / positive,
                                "coverage": int(y.sum()) / positive,
                            }
                            matches = [
                                r
                                for r in records
                                if all(
                                    r[k] == v
                                    for k, v in {
                                        "split": split,
                                        "corpus": corpus,
                                        "seed": seed,
                                        "variant": variant,
                                        "role": role,
                                    }.items()
                                )
                            ]
                            assert len(matches) == 1
                            for key, value in expected.items():
                                error = abs(value - matches[0][key])
                                assert error < 1e-12
                                max_error = max(max_error, error)
                        count += len(values)
    direct = read(run / "score-audit.json")
    assert len(direct) == len(config["encoder_seeds"]) * len(
        config["corpora"]
    ) * 2 * len(VARIANTS)
    write(
        run / "independent-audit.json",
        {
            "training": training,
            "saved_score_values_checked": count,
            "maximum_metric_difference": max_error,
            "maximum_direct_forward_difference": max(
                r["maximum_direct_forward_difference"] for r in direct
            ),
            "initialization_training_samples_and_teacher_updates_verified": True,
            "files": pin(
                [
                    run / "metrics.json",
                    run / "selection-freeze.json",
                    run / "models-freeze.json",
                ]
            ),
        },
        immutable=True,
    )
