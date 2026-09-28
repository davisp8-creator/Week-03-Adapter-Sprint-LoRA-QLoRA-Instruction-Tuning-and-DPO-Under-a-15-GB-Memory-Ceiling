"""
Environment detector + dependency installer for the CivicDesk LoRA/QLoRA project.

Supports two target environments:
  1. Google Colab notebook with a T4 GPU (CUDA available).
  2. Windows Server VM with an Intel CPU (no CUDA GPU).

Usage (run this first, in a notebook cell or as a script):

    import env_setup
    env = env_setup.setup()

`env` is a dict describing what was detected/installed, e.g.:
    {
        "platform": "colab" | "windows" | "other",
        "has_cuda": bool,
        "device": "cuda" | "cpu",
        "gpu_name": str | None,
    }

Running as a script (`python env_setup.py`) does the same and prints a summary.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys

# Package -> pip spec. Version pins kept loose but compatible as of Sep 2025.
BASE_PACKAGES = {
    "transformers": "transformers>=4.44",
    "peft": "peft>=0.12",
    "trl": "trl>=0.9",
    "accelerate": "accelerate>=0.33",
    "datasets": "datasets>=2.20",
    "bitsandbytes": "bitsandbytes>=0.43.1",
}


def detect_environment() -> dict:
    """Identify whether we're on Colab or a local Windows box, and GPU availability."""
    is_colab = False
    try:
        import google.colab  # noqa: F401

        is_colab = True
    except ImportError:
        is_colab = os.environ.get("COLAB_RELEASE_TAG") is not None

    system = platform.system()  # "Windows", "Linux", "Darwin"

    if is_colab:
        plat = "colab"
    elif system == "Windows":
        plat = "windows"
    else:
        plat = "other"

    # nvidia-smi is the most reliable pre-torch signal that a CUDA GPU is present.
    has_nvidia_smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, check=False
    ).returncode == 0 if _command_exists("nvidia-smi") else False

    return {"platform": plat, "system": system, "has_nvidia_smi": has_nvidia_smi}


def _command_exists(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def _pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def _torchao_version() -> str | None:
    """Query the installed torchao version via a fresh subprocess (not this
    process's importlib.metadata cache, which can go stale after a pip
    uninstall/install run from within the same process). Returns None if
    torchao isn't installed/importable."""
    result = subprocess.run(
        [sys.executable, "-c", "import importlib.metadata as m; print(m.version('torchao'))"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _remove_incompatible_torchao() -> None:
    """Colab preinstalls an old torchao (e.g. 0.10.0). peft>=0.21's LoRA dispatch
    calls is_torchao_available(), which *raises* (not returns False) if torchao
    is present but below 0.16.0 -- breaking every PeftModel.from_pretrained/
    get_peft_model call. We don't use torchao (we use bitsandbytes 4-bit)."""
    from packaging.version import parse

    version = _torchao_version()
    if version is None or parse(version) >= parse("0.16.0"):
        return  # absent, or already compatible -- nothing to do

    print(f"Found incompatible torchao=={version} (peft requires >=0.16.0 or absent); removing it.")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])

    remaining = _torchao_version()
    if remaining is None:
        print("torchao removed.")
        return

    # Uninstall didn't stick (seen on some Colab images). Fall back to
    # upgrading it to a version peft accepts, rather than leaving it broken.
    print(f"torchao=={remaining} is still importable after uninstall; trying to upgrade it instead.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"])

    final = _torchao_version()
    if final is not None and parse(final) >= parse("0.16.0"):
        print(f"torchao upgraded to {final}.")
    else:
        print(f"WARNING: could not get torchao to a compatible state (currently: {final}). "
              f"peft's PeftModel.from_pretrained/get_peft_model may still fail with the "
              f"'incompatible version of torchao' ImportError. If it does, restart the Colab "
              f"runtime (Runtime > Restart session) and re-run -- this clears whatever is "
              f"keeping the old torchao resolvable.")


def install_dependencies(env: dict) -> None:
    """Install torch (CPU or CUDA build) plus the shared HF/PEFT/TRL stack."""
    plat = env["platform"]

    if plat == "colab" or env["has_nvidia_smi"]:
        # Colab T4 (or any box with an NVIDIA GPU): CUDA-enabled torch wheel.
        _pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cu121")
    else:
        # Windows Server on Intel CPU: no CUDA, use the CPU-only wheel to
        # avoid pulling multi-GB CUDA runtime deps that won't be used.
        _pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    _pip_install(*BASE_PACKAGES.values())
    _remove_incompatible_torchao()


def resolve_device() -> tuple[str, str | None]:
    """Import torch (after install) and report the device to train/infer on."""
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", None


def setup(install: bool = True) -> dict:
    """Detect environment, install dependencies, and return a summary dict."""
    env = detect_environment()

    if install:
        install_dependencies(env)

    device, gpu_name = resolve_device()
    env.update({"has_cuda": device == "cuda", "device": device, "gpu_name": gpu_name})

    if device == "cpu" and env["platform"] != "colab":
        print(
            "WARNING: no CUDA GPU detected. bitsandbytes 4-bit quantization "
            "(QLoRA) requires a CUDA GPU and will not work here. Use this "
            "environment for CPU-only steps (data prep, small evals) and run "
            "the LoRA/QLoRA + DPO training on the Colab T4 notebook."
        )

    return env


def _print_summary(env: dict) -> None:
    print("Environment summary")
    print(f"  platform   : {env['platform']}")
    print(f"  os         : {env['system']}")
    print(f"  device     : {env['device']}")
    print(f"  gpu        : {env['gpu_name'] or 'none'}")
    for pkg in (*BASE_PACKAGES, "torch"):
        try:
            ver = importlib.import_module(pkg).__version__
        except Exception as exc:  # pragma: no cover - diagnostic only
            ver = f"import failed ({exc})"
        print(f"  {pkg:<12}: {ver}")


if __name__ == "__main__":
    result = setup()
    _print_summary(result)
