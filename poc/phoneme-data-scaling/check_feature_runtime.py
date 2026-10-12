"""Verify saved extraction features in the acoustic training environment."""

import heapq
import wave

from scaling_study import CONFIG, ROOT, fixed, pin, read, s, verify


def main():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    frozen = verify(run)
    prepared = ROOT / config["preparation_run"]
    extract = read(prepared / "extraction-freeze-0.json")
    pipe = s.pipeline(extract["configuration"])
    results = {}
    for group in read(prepared / "preparation-freeze.json")["config"]["groups"]:
        count = read(prepared / "preparation-report.json")["groups"][group]["segments"]
        cache = s.np.memmap(
            prepared / f"{group}-features.f32",
            dtype="<f4",
            mode="r",
            shape=(count, 128),
        )
        probes = heapq.nsmallest(
            200,
            s.rows(prepared / f"{group}-segments.jsonl"),
            key=lambda v: s.rank(2026101206, v["segment_id"]),
        )
        maximum = 0.0
        for row in probes:
            with wave.open(str(ROOT / row["source_file"])) as wav:
                wav.setpos(row["start_frame"])
                pcm = (
                    s.np.frombuffer(
                        wav.readframes(row["end_frame"] - row["start_frame"]),
                        dtype="<i2",
                    ).astype(s.np.float32)
                    / 32768
                )
            features = s.pool_features(pipe, pcm, row["segment_id"])
            maximum = max(
                maximum, float(s.np.abs(features - cache[row["cache_index"]]).max())
            )
        assert maximum <= 2e-6
        results[group] = {
            "count": len(probes),
            "maximum_absolute_difference": maximum,
            "probe_ids": [v["segment_id"] for v in probes],
        }
    fixed(
        run / "feature-runtime-check.json",
        {
            "runtime": frozen["runtime"],
            "groups": results,
            "files": pin([__import__("pathlib").Path(__file__)]),
        },
    )
    print(
        {
            g: {k: v for k, v in d.items() if k != "probe_ids"}
            for g, d in results.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
