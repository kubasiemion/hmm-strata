import torch


def pick() -> str:
    """cuda (Colab GPU), else Intel xpu, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"
