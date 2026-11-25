"""Explicit commands: offline smoke, local artifact analysis, CUDA fit/eval."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from .data import check_heldout, load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Offline fixture math/plumbing only; no model or network")
    smoke.add_argument("--output", required=True)
    handoff_smoke = commands.add_parser("handoff-smoke", help="Offline cache handoff demo")
    handoff_smoke.add_argument("--hf", action="store_true", help="Also verify parity with a tiny CPU model from the hf extra")
    analyze = commands.add_parser("analyze", help="Validate existing traces and create a fresh report")
    analyze.add_argument("--traces", required=True)
    analyze.add_argument("--output", required=True)
    for name in ("fit", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--source", required=True, help="Local safetensors checkpoint directory")
        cmd.add_argument("--target", required=True)
        cmd.add_argument("--data", required=True, help="Strict calibration/eval JSONL")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--execute", action="store_true", help="Required for an approved model run")
        if name == "fit":
            cmd.add_argument("--alpha", type=float, default=.01)
            cmd.add_argument("--prefix-tokens", type=int, default=1024)
        else:
            cmd.add_argument("--mapper", required=True)
            cmd.add_argument("--repeats", type=int, default=1)
            cmd.add_argument("--warmups", type=int, default=1)
            cmd.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.command == "smoke":
        from .smoke import run
        print(json.dumps(run(args.output), indent=2))
        return
    if args.command == "handoff-smoke":
        from .handoff_smoke import run
        print(json.dumps(run(hf=args.hf), indent=2))
        return
    if args.command == "analyze":
        from .evidence import write_report
        rows = [json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()]
        print(json.dumps(write_report(args.output, rows, {"reanalyzed_from": str(Path(args.traces).resolve())}), indent=2))
        return
    if not args.execute:
        print(json.dumps({"status": "plan_only_no_model_loaded", "command": vars(args),
                          "requires": "Per-run approval, CUDA worker, local checkpoints, disjoint data, frozen hashes; see IMPLEMENTATION.md"}, indent=2))
        return
    split = "calibration" if args.command == "fit" else "eval"
    examples = load_examples(args.data, split)
    if Path(args.output).exists():
        raise ValueError("Output must be a new directory")
    if args.command == "evaluate":
        check_heldout(json.loads((Path(args.mapper)/"manifest.json").read_text()), examples)
    from .hf_backend import DenseRuntime, RidgeMapper
    runtime = DenseRuntime.from_local(args.source, args.target, args.device)
    if args.command == "fit":
        mapper = runtime.fit(examples, args.alpha, args.prefix_tokens)
        manifest = mapper.save(args.output)
        manifest["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")}
        (Path(args.output)/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print("Mapper saved:", args.output)
        return
    runtime.mapper = RidgeMapper.load(args.mapper, runtime.identities["source"]["sha256"],
                                      runtime.identities["target"]["sha256"], runtime.tokenizer_sha256)
    runtime.mapper.to(next(runtime.target.parameters()).device)
    check_heldout(runtime.mapper.manifest, examples)
    from .runner import evaluate
    result = evaluate(runtime, examples, args.output, args.repeats, args.warmups, args.max_new_tokens)
    print(json.dumps(result, indent=2))
