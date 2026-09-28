"""
Controller: runs the entire CivicDesk pipeline end to end, on either the
Windows VM or a Colab notebook, and writes the results back to this GitHub
repo.

Stages, in order: environment detection/setup -> SFT dataset generation ->
DPO dataset generation -> QLoRA SFT training (GPU only) -> DPO training (GPU
only, needs the SFT adapter) -> evaluation (GPU if available, else CPU, needs
both adapters). GPU-only stages are *skipped* (not failed) with a clear
reason when there's no CUDA GPU, so this script also runs cleanly, doing just
the CPU-safe stages, on the Windows box.

Each stage runs as its own subprocess (so a training crash can't take down
the controller, and each stage's own --log-every / --mem-limit-gb reporting
still shows up live). Afterwards this script writes a `runs/<timestamp>.json`
record, updates the "Run History" table in README.md, and commits the
results (adapters, logs, data, README) back to the repo.

GitHub push uses a fine-grained personal access token read from the
GITHUB_TOKEN environment variable -- never hardcode it, never commit it. See
README.md "GitHub push access" for how to set it up (Colab Secrets, or a
plain env var on the VM). If GITHUB_TOKEN isn't set, every stage still runs
and results are still committed locally; only the push is skipped.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-eval --no-push
    python scripts/run_pipeline.py --no-commit   # run everything, touch no git state
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(REPO_ROOT))  # so `import env_setup` (repo-root module) works
import env_setup  # noqa: E402

RUN_HISTORY_START = "<!-- RUN_HISTORY:START -->"
RUN_HISTORY_END = "<!-- RUN_HISTORY:END -->"

DEFAULT_HEADER = "| Date | System | GPU | Steps run | Total time | Peak GPU mem |"
DEFAULT_SEP = "|---|---|---|---|---|---|"


def run_stage(name: str, args_list: list[str]) -> dict:
    print(f"\n=== {name} ===")
    print("$", " ".join(args_list))
    start = time.time()
    result = subprocess.run(args_list, cwd=REPO_ROOT)
    elapsed = time.time() - start
    status = "ran" if result.returncode == 0 else "failed"
    print(f"--- {name}: {status} in {elapsed:.1f}s ---")
    return {"name": name, "status": status, "elapsed_sec": round(elapsed, 1)}


def skipped(name: str, reason: str) -> dict:
    print(f"Skipping {name} ({reason}).")
    return {"name": name, "status": f"skipped ({reason})", "elapsed_sec": 0.0}


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, capture_output=True, text=True)


def ensure_git_identity() -> None:
    name = git(["config", "user.name"], check=False).stdout.strip()
    if not name:
        git(["config", "user.name", "CivicDesk Pipeline (automated)"])
        git(["config", "user.email", "pipeline@localhost"])
        print("No git identity configured; using a local automated-run identity for this commit.")


def push_with_token(branch: str) -> None:
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set; skipping push. Results are committed locally -- see README.md "
              "'GitHub push access' for how to set the token up.")
        return

    remote_url = git(["remote", "get-url", "origin"]).stdout.strip()
    if not remote_url.startswith("https://"):
        print(f"origin remote ({remote_url}) isn't an https:// URL; can't push with a token. Skipping push.")
        return
    host_and_path = remote_url[len("https://"):]

    try:
        git(["fetch", "origin", branch])
        git(["merge", "--ff-only", f"origin/{branch}"])
    except subprocess.CalledProcessError:
        print("Could not fast-forward onto origin before pushing (local and remote have diverged). "
              "Resolve manually (git pull), then push yourself. Not pushing automatically.")
        return

    # The token is embedded in this URL only for this one invocation (never written to
    # .git/config, never printed). Deliberately not using the shared git() helper here so
    # a failure's captured stdout/stderr -- which git itself may echo the URL into -- is
    # never touched or printed; only this sanitized message is.
    auth_url = f"https://{token}@{host_and_path}"
    result = subprocess.run(
        ["git", "push", auth_url, f"HEAD:{branch}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Push failed. (Output withheld: it may echo the destination URL, which embeds the token.)")
        return

    print(f"Pushed to origin/{branch}.")


def update_readme(record: dict) -> None:
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        print("No README.md found; skipping automated README update.")
        return
    text = readme_path.read_text(encoding="utf-8")

    if RUN_HISTORY_START not in text or RUN_HISTORY_END not in text:
        print("README.md has no Run History markers; skipping automated README update.")
        return

    before, rest = text.split(RUN_HISTORY_START, 1)
    table_block, after = rest.split(RUN_HISTORY_END, 1)

    lines = [ln for ln in table_block.strip("\n").split("\n") if ln.strip()]
    if len(lines) >= 2:
        header, sep, existing_rows = lines[0], lines[1], lines[2:]
    else:
        header, sep, existing_rows = DEFAULT_HEADER, DEFAULT_SEP, []

    new_row = (
        f"| {record['date']} | {record['system']} | {record['gpu']} | "
        f"{record['steps']} | {record['total_time']} | {record['peak_mem']} |"
    )
    new_block = "\n".join([header, sep, new_row, *existing_rows])

    new_text = f"{before}{RUN_HISTORY_START}\n{new_block}\n{RUN_HISTORY_END}{after}"
    readme_path.write_text(new_text, encoding="utf-8")
    print("Updated README.md Run History table.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-commit", action="store_true", help="run every stage but touch no git state")
    parser.add_argument("--no-push", action="store_true", help="commit locally but don't push")
    parser.add_argument("--branch", default="main")
    # parse_known_args so this also runs unmodified inside Jupyter/Colab, which
    # injects its own "-f <kernel.json>" flag into sys.argv.
    args, _unknown = parser.parse_known_args()

    py = sys.executable
    pipeline_start = time.time()

    print("Detecting environment...")
    env = env_setup.setup(install=True)
    print(f"platform={env['platform']}  os={env['system']}  device={env['device']}  gpu={env['gpu_name']}")

    steps: list[dict] = []

    if not args.skip_data_gen:
        steps.append(run_stage("generate_sft_data", [
            py, str(SCRIPTS_DIR / "generate_synthetic_complaints.py"),
            "--n", "400", "--out", "data/311_complaints.jsonl",
        ]))
        steps.append(run_stage("generate_dpo_data", [
            py, str(SCRIPTS_DIR / "generate_dpo_pairs.py"),
            "--n", "150", "--out", "data/dpo_address_preferences.jsonl",
        ]))
    else:
        steps.append(skipped("generate_sft_data", "skipped by flag"))
        steps.append(skipped("generate_dpo_data", "skipped by flag"))

    sft_adapter = REPO_ROOT / "outputs" / "qlora-gpt2-311" / "adapter"
    dpo_adapter = REPO_ROOT / "outputs" / "dpo-gpt2-311" / "adapter"

    if args.skip_sft:
        steps.append(skipped("train_qlora", "skipped by flag"))
    elif not env["has_cuda"]:
        steps.append(skipped("train_qlora", "no CUDA GPU available"))
    else:
        steps.append(run_stage("train_qlora", [
            py, str(SCRIPTS_DIR / "train_qlora.py"),
            "--data", "data/311_complaints.jsonl",
            "--model-name", args.model_name,
            "--output-dir", "outputs/qlora-gpt2-311",
        ]))

    if args.skip_dpo:
        steps.append(skipped("train_dpo", "skipped by flag"))
    elif not env["has_cuda"]:
        steps.append(skipped("train_dpo", "no CUDA GPU available"))
    elif not sft_adapter.exists():
        steps.append(skipped("train_dpo", "no SFT adapter found"))
    else:
        steps.append(run_stage("train_dpo", [
            py, str(SCRIPTS_DIR / "train_dpo.py"),
            "--sft-adapter", str(sft_adapter),
            "--data", "data/dpo_address_preferences.jsonl",
            "--model-name", args.model_name,
            "--output-dir", "outputs/dpo-gpt2-311",
        ]))

    if args.skip_eval:
        steps.append(skipped("evaluate", "skipped by flag"))
    elif not (sft_adapter.exists() and dpo_adapter.exists()):
        steps.append(skipped("evaluate", "adapters not both present"))
    else:
        steps.append(run_stage("evaluate", [
            py, str(SCRIPTS_DIR / "evaluate_models.py"),
            "--model-name", args.model_name,
            "--sft-adapter", str(sft_adapter),
            "--dpo-adapter", str(dpo_adapter),
            "--out", "outputs/eval_results.jsonl",
        ]))

    total_elapsed = time.time() - pipeline_start

    sft_summary = read_json(REPO_ROOT / "outputs" / "qlora-gpt2-311" / "run_summary.json")
    dpo_summary = read_json(REPO_ROOT / "outputs" / "dpo-gpt2-311" / "run_summary.json")
    peaks = [s["peak_gpu_mem_gb"] for s in (sft_summary, dpo_summary) if s]
    peak_mem = f"{max(peaks):.2f} GB" if peaks else "N/A (no GPU)"

    ran_names = ", ".join(
        s["name"].replace("train_", "").replace("generate_", "gen_") for s in steps if s["status"] == "ran"
    )

    record = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "system": f"{env['platform']} ({env['system']})",
        "gpu": env["gpu_name"] or "CPU only",
        "steps": ran_names or "none",
        "total_time": format_duration(total_elapsed),
        "peak_mem": peak_mem,
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, **record, "steps_detail": steps}, indent=2), encoding="utf-8"
    )

    update_readme(record)

    print("\n=== Pipeline summary ===")
    print(json.dumps(record, indent=2))

    if args.no_commit:
        print("--no-commit set: not touching git.")
        return

    status = git(["status", "--porcelain"]).stdout.strip()
    if not status:
        print("Nothing changed; skipping commit.")
        return

    ensure_git_identity()

    candidate_paths = [
        "README.md", "runs", "data",
        "outputs/qlora-gpt2-311/adapter", "outputs/qlora-gpt2-311/memory_log.jsonl",
        "outputs/qlora-gpt2-311/run_summary.json",
        "outputs/dpo-gpt2-311/adapter", "outputs/dpo-gpt2-311/memory_log.jsonl",
        "outputs/dpo-gpt2-311/run_summary.json", "outputs/eval_results.jsonl",
    ]
    existing_paths = [p for p in candidate_paths if (REPO_ROOT / p).exists()]
    git(["add", *existing_paths])

    commit_message = (
        f"Automated pipeline run ({record['system']}, {record['gpu']}): "
        f"{record['steps']} in {record['total_time']}\n\n"
        f"Generated by scripts/run_pipeline.py -- not a Claude Code interactive commit."
    )
    git(["commit", "-m", commit_message])
    print("Committed results locally.")

    if args.no_push:
        print("--no-push set: commit created locally, not pushed.")
        return

    push_with_token(args.branch)


if __name__ == "__main__":
    main()
