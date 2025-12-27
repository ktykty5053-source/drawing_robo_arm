\
    from __future__ import annotations
    import os, json, random
    from typing import Any, Dict, Optional

    import numpy as np
    import torch

    def set_seed(seed: Optional[int]) -> None:
        """seed가 None이면 비결정적(사용자 선호)"""
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def device_from_arg(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    class RunningMean:
        def __init__(self):
            self.n = 0
            self.v = 0.0
        def update(self, x: float, n: int = 1):
            self.v = (self.v * self.n + x * n) / (self.n + n)
            self.n += n
        @property
        def value(self) -> float:
            return float(self.v)

    def save_jsonl(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def save_checkpoint(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, step: int, best_loss: float, cfg: Dict[str, Any]) -> None:
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "best_loss": best_loss,
            "cfg": cfg,
        }, path)
