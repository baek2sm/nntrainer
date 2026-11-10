#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate two htp_sparse_benefit cohorts and export TTFT CSV/JSON.

Usage: summarize_sparse_benefit.py COLD_JSON CACHE_JSON OUTPUT_STEM
Both input reports must be complete. No external Python dependencies.
"""
import csv
import json
from pathlib import Path
import statistics
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(cold, cache):
    require(cold.get("passed") and cache.get("passed"), "Incomplete experiment")
    require(cold["experiment"] == "cold" and cache["experiment"] == "cache",
            "Expected separate no-prefix and cached cohorts")
    require(cold["cases"] == cache["cases"], "Case fixtures differ")
    require({k: v for k, v in cold["config"].items() if k != "htp"} ==
            {k: v for k, v in cache["config"].items() if k != "htp"},
            "Model settings differ")
    ch, kh = cold["config"]["htp"], cache["config"]["htp"]
    flags = {"kv_session", "kv_paged", "kv_prefix_cache", "kv_sparse_prefix_cache"}
    require(all(not ch[k] and kh[k] for k in flags), "Cache controls differ")
    require({k: v for k, v in ch.items() if k not in flags} ==
            {k: v for k, v in kh.items() if k not in flags},
            "Other HTP controls differ")
    require(not ch["kv_turboquant"] and not kh["kv_turboquant"],
            "TurboQuant must be off")
    groups = {c["name"]: {m: [] for m in ("cold", "exact", "sparse")}
              for c in cold["cases"]}
    for report in (cold, cache):
        seen = set()
        for sample in report["samples"]:
            name, mode = sample["case"], sample["result"]["mode"]
            key = (name, mode, sample["round"])
            require(key not in seen, "Duplicate sample")
            seen.add(key)
            require(mode in (["cold"] if report is cold else ["exact", "sparse"]),
                    "Unexpected cohort mode")
            groups[name][mode].append(sample["result"])
    rows = []
    for case in cold["cases"]:
        row = dict(case)
        group = groups[case["name"]]
        counts = [len(group[m]) for m in ("cold", "exact", "sparse")]
        require(min(counts) >= 2 and len(set(counts)) == 1, "Unbalanced repeats")
        baseline_ids = group["cold"][0]["ids"]
        for mode, samples in group.items():
            require(all(s["flash_hits"] == 0 for s in samples), "Unexpected Flash hit")
            require(all(s["reused_tokens"] + s["prefill_tokens"] ==
                        case["input_tokens"] - 1 for s in samples),
                    "Missing or duplicated prompt rows")
            if mode in ("cold", "exact"):
                require(all(s["sparse_hits"] == 0 for s in samples),
                        "Unexpected sparse path in baseline")
            if mode == "sparse":
                require(all(s["sparse_hits"] == int(
                    s["reused_tokens"] > case["expected_exact_rows"])
                    for s in samples), "Sparse path counter differs from hit")
            if mode == "cold":
                require(all(s["reused_tokens"] == 0 and s["prefix_host_bytes"] == 0
                            for s in samples), "Cold retained/reused prefix KV")
            if mode == "exact":
                require(all(s["reused_tokens"] == case["expected_exact_rows"]
                            for s in samples), "Unexpected exact hit")
            if mode == "sparse":
                require(all(s["reused_tokens"] >= case["expected_sparse_context_rows"]
                            for s in samples), "Incomplete sparse hit")
            def stats(key):
                values = [s[key] for s in samples]
                return {"median": statistics.median(values),
                        "min": min(values), "max": max(values)}
            row[mode] = {
                "n": len(samples),
                "ttft_ms": stats("external_ttft_ms"),
                "model_ttft_ms": stats("ttft_ms"),
                "prefill_ms": stats("prefill_ms"),
                "wall_ms": stats("wall_ms"),
                "reused_tokens": sorted({s["reused_tokens"] for s in samples}),
                "prefill_tokens": sorted({s["prefill_tokens"] for s in samples}),
                "prefix_host_bytes": sorted({s["prefix_host_bytes"] for s in samples}),
                "code_present": all("ORCHID-842" in s["output"] for s in samples),
                "ids_equal_cold": all(s["ids"] == baseline_ids for s in samples),
            }
        row["sparse_vs_cold"] = (row["cold"]["ttft_ms"]["median"] /
                                 row["sparse"]["ttft_ms"]["median"])
        row["sparse_vs_exact"] = (row["exact"]["ttft_ms"]["median"] /
                                  row["sparse"]["ttft_ms"]["median"])
        row["exact_vs_cold"] = (row["cold"]["ttft_ms"]["median"] /
                                row["exact"]["ttft_ms"]["median"])
        rows.append(row)
    return {"metric": "external first-visible-token latency, milliseconds",
            "rows": rows,
            "all_exact_ids_equal_cold": all(r["exact"]["ids_equal_cold"] for r in rows),
            "all_cold_ids_stable": all(r["cold"]["ids_equal_cold"] for r in rows)}


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    reports = [json.loads(Path(p).read_text()) for p in sys.argv[1:3]]
    result = summarize(*reports)
    stem = Path(sys.argv[3])
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["case", "context_tokens", "input_tokens"]
    for mode in ("cold", "exact", "sparse"):
        fields += [f"{mode}_{k}" for k in (
            "ttft_s", "ttft_min_s", "ttft_max_s", "prefill_s", "wall_s",
            "reused_tokens", "fresh_prefill_tokens",
            "prefix_host_bytes", "ids_equal_cold", "code_present")]
    fields += ["sparse_vs_cold", "sparse_vs_exact", "exact_vs_cold"]
    with stem.with_suffix(".csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in result["rows"]:
            row = {"case": r["name"], "context_tokens": r["context_tokens"],
                   "input_tokens": r["input_tokens"]}
            for mode in ("cold", "exact", "sparse"):
                s = r[mode]
                for field, stat in (("ttft_s", "median"), ("ttft_min_s", "min"),
                                    ("ttft_max_s", "max")):
                    row[f"{mode}_{field}"] = round(s["ttft_ms"][stat] / 1000, 4)
                row[f"{mode}_prefill_s"] = round(s["prefill_ms"]["median"] / 1000, 4)
                row[f"{mode}_wall_s"] = round(s["wall_ms"]["median"] / 1000, 4)
                for field, key in (("reused_tokens", "reused_tokens"),
                                   ("fresh_prefill_tokens", "prefill_tokens"),
                                   ("prefix_host_bytes", "prefix_host_bytes")):
                    row[f"{mode}_{field}"] = ";".join(map(str, s[key]))
                for field in ("ids_equal_cold", "code_present"):
                    row[f"{mode}_{field}"] = s[field]
            for field in ("sparse_vs_cold", "sparse_vs_exact", "exact_vs_cold"):
                row[field] = round(r[field], 4)
            writer.writerow(row)
    print("| Case | Context | Exact hit | Sparse hit | Cold s | Exact s | Sparse s | vs cold | vs exact |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in result["rows"]:
        cells = [r["name"], str(r["context_tokens"])]
        cells += ["/".join(map(str, r[m]["reused_tokens"])) for m in ("exact", "sparse")]
        cells += [f'{r[m]["ttft_ms"]["median"] / 1000:.2f}' for m in ("cold", "exact", "sparse")]
        cells += [f'{r[k]:.2f}x' for k in ("sparse_vs_cold", "sparse_vs_exact")]
        print("| " + " | ".join(cells) + " |")
    if not result["all_exact_ids_equal_cold"] or not result["all_cold_ids_stable"]:
        raise SystemExit("Output IDs differ across cold/exact repetitions; see JSON")


if __name__ == "__main__":
    main()
