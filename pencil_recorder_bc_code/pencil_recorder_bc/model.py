\
    from __future__ import annotations
    import torch
    import torch.nn as nn

    class GRUPolicy(nn.Module):
        """
        입력: (x, y, dt, p, pen)  [B, T, D_in]
        출력: (dx, dy, dt_next, dp, pen_next_logit) [B, T, D_out]
        """
        def __init__(self, input_dim: int, hidden: int = 256, layers: int = 2, dropout: float = 0.1, output_dim: int = 5):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.gru(x)
            return self.head(h)
