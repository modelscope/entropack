import torch


def probe() -> str | None:
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False (no NVIDIA device / driver)"
    try:
        import cupy
    except ImportError as e:
        return (f"cupy is not installed ({e}); install a matching build, e.g. "
                "`pip install cupy-cuda13x` for CUDA 13 or `pip install cupy-cuda12x` for CUDA 12")
    return None


def runs_on() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())
