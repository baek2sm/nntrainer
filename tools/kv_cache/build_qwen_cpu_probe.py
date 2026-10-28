#!/usr/bin/env python3
"""Build the logits probe against an existing Meson CPU build, without Ninja rebuild.

Use --header-revision only when observing old binaries while source headers have
changed. A probe must be rebuilt whenever the linked CausalLM ABI changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-kv-cpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = json.loads((build / "compile_commands.json").read_text())
    entry = next(c for c in commands if c["file"].endswith("Applications/CausalLM/main.cpp"))
    argv = shlex.split(entry["command"])
    compile_args = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-o", "-MF", "-MQ", "-c"):
            i += 2
        elif argv[i] == "-MD":
            i += 1
        else:
            compile_args.append(argv[i])
            i += 1
    if args.header_revision:
        include = output.parent / "baseline-headers"
        include.mkdir(exist_ok=True)
        for path in ("Applications/CausalLM/models/transformer.h",
                     "Applications/CausalLM/models/causal_lm.h",
                     "Applications/CausalLM/kv_cache_manager.h"):
            content = subprocess.check_output([
                "git", "show", f"{args.header_revision}:{path}"], cwd=root)
            (include / Path(path).name).write_bytes(content)
        compile_args.insert(1, "-I" + str(include))
    obj = output.with_suffix(".o")
    compile_args += ["-c", str(root / "tools/kv_cache/qwen_cpu_probe.cpp"), "-o", str(obj)]
    subprocess.run(compile_args, cwd=build, check=True)
    link = subprocess.check_output([
        "ninja", "-t", "commands", "Applications/CausalLM/nntr_causallm"], cwd=build, text=True).splitlines()[-1]
    link_args = shlex.split(link)
    i = link_args.index("-o")
    link_args[i + 1] = str(output)
    link_args = [str(obj) if a.endswith("_Applications_CausalLM_main.cpp.o") else a
                 for a in link_args]
    # The original executable's relative RUNPATH assumes Applications/CausalLM.
    link_args += ["-Wl,-rpath," + str(build / "Applications/CausalLM"),
                  "-Wl,-rpath," + str(build / "nntrainer")]
    subprocess.run(link_args, cwd=build, check=True)
    metadata = {"header_revision": args.header_revision,
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_tree_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root)),
                "source_diff_sha256": hashlib.sha256(subprocess.check_output(
                    ["git", "diff"], cwd=root)).hexdigest(),
                "probe_source_sha256": hashlib.sha256(
                    (root / "tools/kv_cache/qwen_cpu_probe.cpp").read_bytes()).hexdigest(),
                "compile_command": compile_args, "link_command": link_args}
    output.with_suffix(".abi.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
