\
    from __future__ import annotations
    import os, json, glob
    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional, Tuple

    import numpy as np
    import torch
    from torch.utils.data import Dataset

    def _get(d: Dict[str, Any], keys: List[str], default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    def _as_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _extract_strokes(obj: Any) -> List[Any]:
        if isinstance(obj, dict):
            if "strokes" in obj and isinstance(obj["strokes"], list):
                return obj["strokes"]
            for k in ("data", "session", "recording", "payload"):
                if k in obj and isinstance(obj[k], dict):
                    s = _extract_strokes(obj[k])
                    if s:
                        return s
        return []

    def _stroke_points(stroke: Any) -> List[Dict[str, Any]]:
        if isinstance(stroke, list):
            return [p for p in stroke if isinstance(p, dict)]
        if isinstance(stroke, dict):
            pts = _get(stroke, ["points", "pts", "path", "samples"], default=None)
            if isinstance(pts, list):
                return [p for p in pts if isinstance(p, dict)]
        return []

    def load_session_json(path: str) -> Tuple[Dict[str, Any], List[List[Dict[str, Any]]]]:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        meta = {}
        if isinstance(obj, dict):
            meta = _get(obj, ["meta", "metadata"], default={}) or {}
        strokes_raw = _extract_strokes(obj)

        strokes: List[List[Dict[str, Any]]] = []
        for s in strokes_raw:
            pts = _stroke_points(s)
            if len(pts) >= 1:
                strokes.append(pts)

        # meta 보강: width/height 추론
        width = None
        height = None
        if isinstance(meta, dict):
            width = _get(meta, ["width", "w", "canvasWidth", "canvas_width"], None)
            height = _get(meta, ["height", "h", "canvasHeight", "canvas_height"], None)
        if width is None or height is None:
            xs, ys = [], []
            for st in strokes:
                for p in st:
                    xs.append(_as_float(_get(p, ["x", "clientX", "cx"], 0.0)))
                    ys.append(_as_float(_get(p, ["y", "clientY", "cy"], 0.0)))
            if xs and ys:
                width = max(xs) if width is None else width
                height = max(ys) if height is None else height
        meta = dict(meta) if isinstance(meta, dict) else {}
        if width is not None: meta["width"] = float(width)
        if height is not None: meta["height"] = float(height)

        return meta, strokes

    @dataclass
    class BuildOptions:
        normalize: bool = True
        use_pressure: bool = True
        dt_clip: float = 0.5  # seconds

    def build_sequence(meta: Dict[str, Any], strokes: List[List[Dict[str, Any]]], opt: BuildOptions) -> np.ndarray:
        """
        반환: [N, D] feature sequence
        feature = [x, y, dt, p, pen] (use_pressure=False면 p 제외 → [x,y,dt,pen])
        pen: 1(펜 닿음) / 0(펜 듦)
        stroke 사이에는 pen=0 토큰 1개를 삽입
        """
        W = float(meta.get("width", 1.0) or 1.0)
        H = float(meta.get("height", 1.0) or 1.0)
        W = max(W, 1.0)
        H = max(H, 1.0)

        seq = []
        prev_t = None

        def norm_xy(x, y):
            if not opt.normalize:
                return x, y
            return x / W, y / H

        for si, st in enumerate(strokes):
            if si > 0 and len(seq) > 0:
                last = seq[-1]
                x0, y0 = last[0], last[1]
                dt0 = 0.0
                if opt.use_pressure:
                    seq.append([x0, y0, dt0, 0.0, 0.0])
                else:
                    seq.append([x0, y0, dt0, 0.0])

            for p in st:
                x = _as_float(_get(p, ["x", "clientX", "cx"], 0.0))
                y = _as_float(_get(p, ["y", "clientY", "cy"], 0.0))
                t = _get(p, ["t", "time", "timestamp", "ms"], None)
                t = None if t is None else _as_float(t, None)

                if prev_t is None or t is None:
                    dt = 0.0
                else:
                    raw_dt = t - prev_t
                    if raw_dt > 10.0:  # ms로 추정
                        raw_dt = raw_dt / 1000.0
                    dt = float(raw_dt)
                dt = float(np.clip(dt, 0.0, opt.dt_clip))

                pr = _get(p, ["p", "pressure", "force"], 0.0)
                pr = float(np.clip(_as_float(pr, 0.0), 0.0, 1.0))

                x, y = norm_xy(x, y)

                if opt.use_pressure:
                    seq.append([x, y, dt, pr, 1.0])
                else:
                    seq.append([x, y, dt, 1.0])

                prev_t = t if t is not None else prev_t

        return np.asarray(seq, dtype=np.float32)

    class StrokeBCDataset(Dataset):
        """
        window 길이 = seq_len+1
        input  = window[:-1]
        target = (dx, dy, dt_next, dp, pen_next) or (dx,dy,dt_next,pen_next)
        """
        def __init__(self, data_dir: str, seq_len: int = 128, normalize: bool = True, use_pressure: bool = True, dt_clip: float = 0.5, max_files: Optional[int] = None):
            self.data_dir = data_dir
            self.seq_len = int(seq_len)
            self.opt = BuildOptions(normalize=normalize, use_pressure=use_pressure, dt_clip=float(dt_clip))

            files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
            if max_files is not None:
                files = files[: int(max_files)]
            if len(files) == 0:
                raise FileNotFoundError(f"No .json files found in: {data_dir}")

            self.seqs: List[np.ndarray] = []
            self.index: List[Tuple[int, int]] = []
            for fp in files:
                meta, strokes = load_session_json(fp)
                if len(strokes) == 0:
                    continue
                seq = build_sequence(meta, strokes, self.opt)
                if seq.shape[0] < self.seq_len + 1:
                    continue
                sid = len(self.seqs)
                self.seqs.append(seq)
                for start in range(0, seq.shape[0] - (self.seq_len + 1) + 1):
                    self.index.append((sid, start))

            if len(self.index) == 0:
                raise RuntimeError("Not enough data to create any training windows. (Need longer sequences or smaller seq_len)")

            self.input_dim = self.seqs[0].shape[1]
            self.output_dim = 5 if use_pressure else 4

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int):
            sid, start = self.index[idx]
            seq = self.seqs[sid][start : start + self.seq_len + 1]  # [L+1, D]
            x = seq[:-1]
            nxt = seq[1:]

            dxdy = nxt[:, 0:2] - x[:, 0:2]
            dt_next = nxt[:, 2:3]
            if self.opt.use_pressure:
                dp = (nxt[:, 3:4] - x[:, 3:4])
                pen_next = nxt[:, 4:5]
                y = np.concatenate([dxdy, dt_next, dp, pen_next], axis=1)  # [L,5]
            else:
                pen_next = nxt[:, 3:4]
                y = np.concatenate([dxdy, dt_next, pen_next], axis=1)  # [L,4]

            return torch.from_numpy(x), torch.from_numpy(y)

    def collate_batch(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
