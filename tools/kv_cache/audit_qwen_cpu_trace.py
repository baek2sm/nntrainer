#!/usr/bin/env python3
"""Check the CPU trace witnesses against the declared Qwen cache allocations.

This checks instrumented application operations, not hardware memory traffic.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def audit(path):
    text = path.read_text()
    headers = re.findall(r"\[kv-cpu-storage\] backing_bytes=(\d+) owner_layers=(\d+)", text)
    if not headers:
        raise ValueError(f"No core CPU storage witness in {path}")
    allocations = re.findall(
        r"\[kv-cpu-storage\] layer=(\d+) key=(0x[0-9a-f]+) value=(0x[0-9a-f]+) "
        r"plane_bytes=(\d+) source=(-?\d+)", text)
    pairs = {(key, value): int(size) for _, key, value, size, _ in allocations}
    owners = {(key, value): int(size) for _, key, value, size, source in allocations
              if int(source) == -1}
    declared_bytes, declared_owners = map(int, headers[-1])
    if len(owners) != declared_owners or sum(owners.values()) * 2 != declared_bytes:
        raise ValueError(f"Storage byte or owner count mismatch in {path}")
    accesses = re.findall(
        r"\[kv-cpu-access\] from=(\d+) tokens=(\d+) key=(0x[0-9a-f]+) "
        r"value=(0x[0-9a-f]+) read_alias=(\d+) write_alias=(\d+) "
        r"append_copy_bytes=(\d+) transform_bytes=(\d+)", text)
    reads = re.findall(
        r"\[kv-cpu-read\] consumer=([^ ]+) history_copy_bytes=(\d+)", text)
    if not accesses or len(reads) != len(accesses):
        raise ValueError(f"Missing producer or consumer witnesses in {path}")
    groups = defaultdict(Counter)
    for position, tokens, key, value, read_alias, write_alias, copy, transform in accesses:
        if (key, value) not in pairs or read_alias != "1" or write_alias != "1" or copy != "0":
            raise ValueError(f"Cache alias or append copy mismatch in {path}")
        if int(position) + int(tokens) > 512 or int(transform) != int(tokens) * 1024 * 2 * 2:
            raise ValueError(f"Unexpected Qwen geometry or transform byte count in {path}")
        groups[(position, tokens)][(key, value)] += 1
    for group in groups.values():
        if set(group) != set(owners) or any(count != 1 for count in group.values()):
            raise ValueError(f"A step is missing or duplicating a KV owner in {path}")
    if "profile=compatibility" in text or re.search(
            r"\[kv-cpu-layout\].*history_copy_bytes=[1-9][0-9]*", text):
        raise ValueError(f"A compatibility or layout history-copy path ran in {path}")
    if any(copy != "0" or arm not in ("flash-strided", "per-row") for arm, copy in reads):
        raise ValueError(f"Unsupported consumer or history copy in {path}")
    return {"file": str(path), "backing_bytes": declared_bytes,
            "owner_layers": declared_owners, "producer_calls": len(accesses),
            "consumers": sorted({arm for arm, _ in reads}),
            "allocation_identity_matches": True,
            "read_alias": True, "write_alias": True,
            "append_copy_bytes": 0, "history_copy_bytes": 0,
            "transform_bytes": sum(int(row[-1]) for row in accesses), "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.stderr.txt"))
    report = {"profile": "Qwen3-0.6B, capacity512, kv_width1024, half-bit storage",
              "runs": [audit(path) for path in files]}
    if not files:
        parser.error("No probe stderr traces found")
    report["passed"] = all(row["passed"] for row in report["runs"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(files)} CPU trace files")


if __name__ == "__main__":
    main()
