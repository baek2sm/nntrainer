#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# @file   test_build_act_scales.py
# @brief  Unit tests for build_act_scales.py scale-group union-find (spec 4.3).
# @author Seungbaek Hong <sb92.hong@samsung.com>
#
# Run: python3 test_build_act_scales.py   (no framework dep, x86)

import build_act_scales as B

MARGIN, I8 = B.MARGIN, B.INT8_MAX


def s(amax):
    return MARGIN * amax / I8


def approx(a, b, eps=1e-9):
    return abs(a - b) <= eps + eps * abs(b)


def test_concat_unions_inputs_and_output():
    topo = B.parse_topo([
        "a\tconv2d\t",
        "b\tconv2d\t",
        "c\tconcat\ta,b",
    ])
    post, pre = B.parse_amax([["a\t10\n", "b\t40\n", "c\t40\n"]])
    scales, warns = B.build_scale_table(topo, post, pre)
    # all three share the group max amax = 40
    assert approx(scales["a"], s(40)), scales["a"]
    assert approx(scales["b"], s(40))
    assert approx(scales["c"], s(40))
    # 'a' own amax 10 inflated to 40 -> ratio 4.0, not > 4.0, no warn
    assert warns == [], warns


def test_concat_warns_on_inflation():
    topo = B.parse_topo(["a\tconv2d\t", "b\tconv2d\t", "c\tconcat\ta,b"])
    post, pre = B.parse_amax([["a\t5\n", "b\t100\n", "c\t100\n"]])
    scales, warns = B.build_scale_table(topo, post, pre)
    assert approx(scales["a"], s(100))
    assert any("a" in w for w in warns), warns  # 100/5 = 20x > 4x


def test_addition_unions_inputs_but_not_output():
    # residual: add output kept separate (spec 5.6)
    topo = B.parse_topo([
        "x\tconv2d\t",
        "y\tconv2d\t",
        "add\taddition\tx,y",
    ])
    post, pre = B.parse_amax([["x\t8\n", "y\t20\n", "add\t50\n"]])
    scales, _ = B.build_scale_table(topo, post, pre)
    assert approx(scales["x"], s(20))   # inputs unioned -> max 20
    assert approx(scales["y"], s(20))
    assert approx(scales["add"], s(50)) # output independent


def test_passthrough_shares_input_scale():
    for t in ("slice", "upsample2d", "pooling2d", "multiout"):
        topo = B.parse_topo(["p\tconv2d\t", "q\t%s\tp" % t])
        post, pre = B.parse_amax([["p\t30\n", "q\t12\n"]])
        scales, _ = B.build_scale_table(topo, post, pre)
        assert approx(scales["q"], s(30)), (t, scales["q"])  # follows producer
        assert approx(scales["p"], s(30))


def test_conv_and_attention_stay_isolated():
    topo = B.parse_topo(["p\tconv2d\t", "psa\tpsa_attention\tp", "c\tconv2d\tpsa"])
    post, pre = B.parse_amax([["p\t30\n", "psa\t7\n", "c\t3\n"]])
    scales, _ = B.build_scale_table(topo, post, pre)
    assert approx(scales["p"], s(30))
    assert approx(scales["psa"], s(7))   # not unioned with p
    assert approx(scales["c"], s(3))


def test_preact_and_outN_keys():
    topo = B.parse_topo(["k/conv\tconv2d\t", "k/conv/generated_out_0\tmultiout\tk/conv"])
    post, pre = B.parse_amax([[
        "k/conv:preact\t38\n",
        "k/conv\t34\n",
        "k/conv/generated_out_0:out0\t34\n",
        "k/conv/generated_out_0:out1\t34\n",
    ]])
    scales, _ = B.build_scale_table(topo, post, pre)
    assert approx(scales["k/conv"], s(34))
    assert approx(scales["k/conv/generated_out_0"], s(34))  # multiout follows conv
    assert approx(scales["k/conv:preact"], s(38))           # preact separate


def test_multi_file_takes_per_key_max():
    topo = B.parse_topo(["a\tconv2d\t"])
    post, pre = B.parse_amax([["a\t10\n"], ["a\t25\n"], ["a\t7\n"]])
    scales, _ = B.build_scale_table(topo, post, pre)
    assert approx(scales["a"], s(25))


def test_chain_transitive_union():
    # conv -> multiout -> slice -> concat(with other conv): all share group max
    topo = B.parse_topo([
        "cv\tconv2d\t",
        "mo\tmultiout\tcv",
        "sl\tslice\tmo",
        "other\tconv2d\t",
        "cat\tconcat\tsl,other",
    ])
    post, pre = B.parse_amax([[
        "cv\t9\n", "mo\t9\n", "sl\t6\n", "other\t44\n", "cat\t44\n",
    ]])
    scales, _ = B.build_scale_table(topo, post, pre)
    for n in ("cv", "mo", "sl", "other", "cat"):
        assert approx(scales[n], s(44)), (n, scales[n])


def test_input_edge_scale_emitted():
    # conv 'c' consumes multiout 'mo' (== producer 'cv' scale); c:in == cv scale.
    topo = B.parse_topo([
        "cv\tconv2d\t",
        "mo\tmultiout\tcv",
        "c\tconv2d\tmo",
    ])
    post, pre = B.parse_amax([["cv\t20\n", "mo\t20\n", "c\t5\n"]])
    scales, _ = B.build_scale_table(topo, post, pre)
    assert approx(scales["c:in"], s(20)), scales.get("c:in")  # producer edge
    assert approx(scales["mo:in"], s(20))                     # multiout input
    assert "cv:in" not in scales                              # source has no input


def test_input_edge_scale_multi_input_takes_max():
    topo = B.parse_topo(["x\tconv2d\t", "y\tconv2d\t", "cat\tconcat\tx,y"])
    post, pre = B.parse_amax([["x\t8\n", "y\t30\n", "cat\t30\n"]])
    scales, _ = B.build_scale_table(topo, post, pre)
    # concat unions inputs+output to 30, so cat:in == 30
    assert approx(scales["cat:in"], s(30))


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS " + t.__name__)
    print("OK (%d tests)" % len(tests))


if __name__ == "__main__":
    _run()
