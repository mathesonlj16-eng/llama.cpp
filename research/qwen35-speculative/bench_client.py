#!/usr/bin/env python3
"""Reproducible localhost-only llama-server completion benchmark.

This measures the server's reported generation rate and the client's
end-to-end request latency. It does NOT measure CUDA checkpoint costs;
collect those separately with Nsight Systems.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROMPTS = [
    "Explain how CPU-to-GPU memory transfers can limit inference speed in three sentences.",
    "Write readable pseudocode for binary search on a sorted array.",
    "Describe four ways a benchmark can produce misleading results.",
]

def localhost_api(base):
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only http://localhost or http://127.0.0.1 is allowed")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a plain local server URL without credentials/query")
    return base.rstrip("/") + "/completion"

def complete(api, prompt, n_predict, timeout):
    data = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "seed": 424242, "cache_prompt": False, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(api, data=data, headers={"Content-Type": "application/json"})
    begin = time.perf_counter_ns()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    elapsed_ms = (time.perf_counter_ns() - begin) / 1_000_000
    if not isinstance(result, dict) or "error" in result:
        raise RuntimeError("Unexpected /completion response: " + repr(result)[:500])
    text = result.get("content")
    if not isinstance(text, str):
        raise RuntimeError("Completion response lacked text content")
    timings = result.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    speed = timings.get("predicted_per_second")
    if speed is not None and (not isinstance(speed, (int, float)) or speed < 0):
        raise RuntimeError("Invalid predicted_per_second field")
    return {
        "client_wall_ms": round(elapsed_ms, 3),
        "server_predicted_tokens_per_second": speed,
        "server_timings": timings,
        "tokens_predicted": result.get("tokens_predicted"),
        "content_chars": len(text),
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "stopped_eos": result.get("stopped_eos"),
        "stopped_limit": result.get("stopped_limit"),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--variant", required=True,
                   help="e.g. vanilla, spec-default, spec-experimental")
    p.add_argument("--build", required=True, help="exact llama.cpp commit or build identifier")
    p.add_argument("--session", required=True, help="unique session name or number")
    p.add_argument("--notes", default="", help="include relevant server flags and model hashes")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = p.parse_args()

    for value in (args.variant, args.session):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            p.error("--variant and --session may contain only letters, digits, _ or -")
    if min(args.rounds, args.n_predict) < 1 or args.warmups < 0 or args.timeout <= 0:
        p.error("rounds, n-predict, timeout must be > 0; warmups must be >= 0")
    try:
        api = localhost_api(args.url)
    except ValueError as exc:
        p.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.output_dir / f"{created}-{args.variant}-{args.session}.jsonl"
    rng = random.Random(20261123)
    samples = []
    with path.open("x", encoding="utf-8") as log:
        meta = {
            "kind": "metadata", "created_utc": created, "variant": args.variant,
            "build": args.build, "session": args.session, "notes": args.notes,
            "url": args.url, "rounds": args.rounds, "warmups": args.warmups,
            "n_predict": args.n_predict, "prompts": PROMPTS,
        }
        log.write(json.dumps(meta, sort_keys=True) + "\n")
        log.flush()
        print(f"Writing {path}", flush=True)
        for w in range(args.warmups):
            complete(api, PROMPTS[w % len(PROMPTS)], args.n_predict, args.timeout)
        for round_idx in range(args.rounds):
            indices = list(range(len(PROMPTS)))
            rng.shuffle(indices)
            for idx in indices:
                try:
                    measurement = complete(api, PROMPTS[idx], args.n_predict, args.timeout)
                except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                    print(f"Stopped after {len(samples)} samples: {exc}", file=sys.stderr)
                    return 2
                row = {"kind": "sample", "round": round_idx + 1,
                       "prompt_index": idx, **measurement}
                samples.append(row)
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                log.flush()

    rates = [s["server_predicted_tokens_per_second"] for s in samples
             if s["server_predicted_tokens_per_second"] is not None]
    print(json.dumps({
        "output": str(path), "samples": len(samples),
        "server_rate_median_tok_s": round(statistics.median(rates), 3) if rates else None,
        "client_wall_median_ms": round(statistics.median(s["client_wall_ms"] for s in samples), 3)
    }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
