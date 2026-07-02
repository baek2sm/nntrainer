#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# @file   build_act_scales.py
# @brief  Offline converter: calibration amax + graph topology -> per-tensor
#         activation scale table for the YOLOv11 W4A8 static-Q8_0 path.
# @details Implements spec section 4.3/4.4 (yolov11_w4a8_static_q8_spec.md):
#          scale-group union-find over concat/add/slice/upsample/pool/multiout
#          edges, then s = margin * max(group amax) / 127. Emits a flat
#          <model>.act_scales.json consumed by the YOLOv11 app graph loader.
#          Pure offline tooling (no device, no framework coupling).
#
# @author Seungbaek Hong <sb92.hong@samsung.com>
#
# Usage:
#   python3 build_act_scales.py --topo topo.tsv --amax calib1.tsv [calib2.tsv ...] \
#           --out yolov11m.act_scales.json
#
# Inputs (produced on-device, env-gated, see neuralnet.cpp / conv2d_layer.cpp):
#   topo.tsv : one line per node  "<name>\t<type>\t<in0,in1,...>"  (NNTR_CALIB_TOPO)
#   *.tsv    : one line per obs    "<key>\t<amax>"                  (NNTR_CALIB_DUMP)
#              key = "<node>"            post-act edge amax
#                    "<node>:outN"       multiout fan-out copy (== producer)
#                    "<node>:preact"     fused-conv pre-activation amax

import argparse
import json
import sys

# Quantization margin (spec 4.1): s = MARGIN * max_amax / 127, sat8 clamp [-127,127].
MARGIN = 1.05
INT8_MAX = 127.0
# group_s / own_s above this loses >2 effective bits -> flag as gate-A suspect.
WARN_RATIO = 4.0

# Node types whose output preserves / subsets the input value range, so the
# output edge shares one scale with its input (spec 4.3 steps 4 + multiout).
PASSTHROUGH_TYPES = {"slice", "upsample2d", "pooling2d", "multiout"}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_topo(lines):
    """lines -> {node: (type, [input_node,...])}. Blank/short lines ignored."""
    topo = {}
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        name, ntype = parts[0], (parts[1] if len(parts) > 1 else "")
        conns = parts[2].split(",") if len(parts) > 2 and parts[2] else []
        topo[name] = (ntype, [c for c in conns if c])
    return topo


def _node_of(key):
    """Map an amax key to its owning graph node (strip :preact / :outN)."""
    if key.endswith(":preact"):
        return key[: -len(":preact")], "preact"
    if ":out" in key:
        return key.rsplit(":out", 1)[0], "post"
    return key, "post"


def parse_amax(line_sets):
    """Multiple calibration files -> (post_amax{node:max}, preact_amax{node:max})."""
    post, preact = {}, {}
    for lines in line_sets:
        for raw in lines:
            line = raw.rstrip("\n")
            if not line:
                continue
            key, _, val = line.partition("\t")
            if not val:
                continue
            try:
                amax = float(val)
            except ValueError:
                continue
            node, kind = _node_of(key)
            tgt = preact if kind == "preact" else post
            if amax > tgt.get(node, 0.0):
                tgt[node] = amax
    return post, preact


def build_scale_table(topo, post_amax, preact_amax):
    """Core (spec 4.3/4.4). Returns (scales{name:s | name:preact:s}, warnings[])."""
    uf = UnionFind()
    for node in topo:  # every node starts in its own group
        uf.find(node)

    for node, (ntype, ins) in topo.items():
        if ntype == "concat":
            for i in ins:  # union all inputs with the concat output
                uf.union(node, i)
        elif ntype == "addition":
            for i in ins:  # union inputs only; output kept separate (spec 5.6)
                uf.union(ins[0], i)
        elif ntype in PASSTHROUGH_TYPES:
            for i in ins:  # value-preserving: output shares input scale
                uf.union(node, i)
        # conv2d / input / psa_attention / add-output: own group, no union.

    # group amax = max over member post-act amax (nodes with no obs contribute 0).
    group_amax = {}
    for node in topo:
        root = uf.find(node)
        a = post_amax.get(node, 0.0)
        if a > group_amax.get(root, 0.0):
            group_amax[root] = a

    def scale_of(amax):
        return MARGIN * amax / INT8_MAX if amax > 0.0 else 0.0

    scales, warnings = {}, []
    for node in topo:
        root = uf.find(node)
        g_amax = group_amax.get(root, 0.0)
        scales[node] = scale_of(g_amax)
        own = post_amax.get(node, 0.0)
        if own > 0.0 and g_amax > WARN_RATIO * own:
            warnings.append(
                "group scale inflates %s: own_amax=%.5g group_amax=%.5g (x%.1f)"
                % (node, own, g_amax, g_amax / own)
            )

    for node, amax in preact_amax.items():  # fused-conv pre-activation scales
        scales["%s:preact" % node] = scale_of(amax)

    return scales, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build W4A8 activation scale table.")
    ap.add_argument("--topo", required=True, help="graph topology tsv (NNTR_CALIB_TOPO)")
    ap.add_argument("--amax", required=True, nargs="+", help="amax tsv(s) (NNTR_CALIB_DUMP)")
    ap.add_argument("--out", required=True, help="output act_scales.json")
    args = ap.parse_args(argv)

    with open(args.topo) as f:
        topo = parse_topo(f.readlines())
    amax_sets = []
    for path in args.amax:
        with open(path) as f:
            amax_sets.append(f.readlines())
    post_amax, preact_amax = parse_amax(amax_sets)

    scales, warnings = build_scale_table(topo, post_amax, preact_amax)

    with open(args.out, "w") as f:
        json.dump(scales, f, indent=1, sort_keys=True)
        f.write("\n")

    n_zero = sum(1 for v in scales.values() if v == 0.0)
    print("nodes=%d  scales=%d (zero=%d)  preact=%d  warnings=%d"
          % (len(topo), len(scales), n_zero, len(preact_amax), len(warnings)))
    for w in warnings:
        print("  WARN " + w, file=sys.stderr)
    print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
