"""Parallel sweep driver: distribute configs across GPUs via subprocess.

Usage:
    python run_sweep.py <sweep_name> [--gpus 0,1] [--out-dir results]

Each config is launched as its own subprocess with CUDA_VISIBLE_DEVICES set
to one physical GPU. At most N_GPUs configs run concurrently; when a worker
finishes, it grabs the next config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = Path(__file__).resolve().parent.parent

from sweeps import SWEEPS


def _cfg_to_args(cfg: dict, out_dir: Path) -> list[str]:
    args = [sys.executable, str(Path(__file__).parent / 'run_single.py'), '--out-dir', str(out_dir)]
    for k, v in cfg.items():
        args.extend([f'--{k.replace("_", "-")}', str(v)])
    return args


def _worker(gpu_id: int, queue: Queue, out_dir: Path, log_dir: Path, results: list):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            return

        idx, total, cfg = item
        name = cfg['name']
        json_path = out_dir / f'{name}.json'
        # Skip only if a previous run FINISHED. Interrupted runs (partial CSV,
        # completed=False in JSON, or no JSON at all) are re-run fresh.
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                meta = {}
            if meta.get('completed') is True:
                print(f"[gpu{gpu_id}] SKIP {name} (already completed)", flush=True)
                results.append((name, 0, 'skipped'))
                queue.task_done()
                continue

        print(f"[gpu{gpu_id}] ({idx}/{total}) START {name}", flush=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')

        log_path = log_dir / f'{name}.log'
        start = time.time()
        with open(log_path, 'w') as logf:
            proc = subprocess.run(_cfg_to_args(cfg, out_dir), env=env, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.time() - start

        status = 'OK' if proc.returncode == 0 else f'FAIL rc={proc.returncode}'
        print(f"[gpu{gpu_id}] ({idx}/{total}) {status} {name} in {elapsed:.0f}s -> {log_path}", flush=True)
        results.append((name, elapsed, status))
        queue.task_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep', choices=list(SWEEPS.keys()))
    p.add_argument('--gpus', default='0,1', help='comma-separated GPU ids')
    p.add_argument('--out-dir', default=None, help='defaults to results/<sweep>')
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(',') if g.strip() != '']
    configs = SWEEPS[args.sweep]
    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / 'results' / args.sweep
    log_dir = out_dir / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep '{args.sweep}': {len(configs)} configs across GPUs {gpus}")
    print(f"Output dir: {out_dir}\n")

    queue: Queue = Queue()
    for i, cfg in enumerate(configs, 1):
        queue.put((i, len(configs), cfg))
    for _ in gpus:
        queue.put(None)

    results: list = []
    threads = [
        threading.Thread(target=_worker, args=(gpu, queue, out_dir, log_dir, results), daemon=True)
        for gpu in gpus
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    for name, elapsed, status in results:
        print(f"  {status:>8}  {elapsed:7.0f}s  {name}")
    failed = [r for r in results if r[2].startswith('FAIL')]
    print(f"\nTotal: {len(results)} runs, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
