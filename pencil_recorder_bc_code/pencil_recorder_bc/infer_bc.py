\
    from __future__ import annotations
    import json, argparse
    import numpy as np
    import torch

    from model import GRUPolicy
    from dataset import BuildOptions, load_session_json, build_sequence
    from utils import device_from_arg

    def sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--json_path", required=True, help="한 세션 json 파일")
        ap.add_argument("--out_path", required=True, help="생성된 시퀀스를 저장할 json")
        ap.add_argument("--steps", type=int, default=1000)
        ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
        ap.add_argument("--use_pressure", type=int, default=1)
        ap.add_argument("--normalize", type=int, default=1)
        ap.add_argument("--dt_clip", type=float, default=0.5)
        args = ap.parse_args()

        device = device_from_arg(args.device)
        meta, strokes = load_session_json(args.json_path)
        opt = BuildOptions(normalize=bool(args.normalize), use_pressure=bool(args.use_pressure), dt_clip=args.dt_clip)
        seq = build_sequence(meta, strokes, opt)
        if seq.shape[0] < 10:
            raise RuntimeError("session too short")

        input_dim = seq.shape[1]
        output_dim = 5 if opt.use_pressure else 4

        model = GRUPolicy(input_dim=input_dim, output_dim=output_dim)
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        T = min(128, seq.shape[0])
        hist = seq[-T:].copy()
        out_seq = [hist[i].tolist() for i in range(hist.shape[0])]

        with torch.no_grad():
            for _ in range(args.steps):
                x = torch.from_numpy(hist[None, :, :]).to(device)
                pred = model(x)[0, -1].cpu().numpy()

                if output_dim == 5:
                    dx, dy, dt_next, dp, pen_logit = pred.tolist()
                    pen_next = 1.0 if sigmoid(pen_logit) > 0.5 else 0.0
                    nxt = hist[-1].copy()
                    nxt[0] = float(np.clip(nxt[0] + dx, 0.0, 1.0 if opt.normalize else 1e9))
                    nxt[1] = float(np.clip(nxt[1] + dy, 0.0, 1.0 if opt.normalize else 1e9))
                    nxt[2] = float(np.clip(dt_next, 0.0, opt.dt_clip))
                    nxt[3] = float(np.clip(nxt[3] + dp, 0.0, 1.0))
                    nxt[4] = float(pen_next)
                else:
                    dx, dy, dt_next, pen_logit = pred.tolist()
                    pen_next = 1.0 if sigmoid(pen_logit) > 0.5 else 0.0
                    nxt = hist[-1].copy()
                    nxt[0] = float(np.clip(nxt[0] + dx, 0.0, 1.0 if opt.normalize else 1e9))
                    nxt[1] = float(np.clip(nxt[1] + dy, 0.0, 1.0 if opt.normalize else 1e9))
                    nxt[2] = float(np.clip(dt_next, 0.0, opt.dt_clip))
                    nxt[3] = float(pen_next)

                out_seq.append(nxt.tolist())
                if hist.shape[0] >= 128:
                    hist = np.concatenate([hist[1:], nxt[None, :]], axis=0)
                else:
                    hist = np.concatenate([hist, nxt[None, :]], axis=0)

        out = {
            "meta": {
                "source": args.json_path,
                "normalize": bool(args.normalize),
                "use_pressure": bool(args.use_pressure),
                "dt_clip": args.dt_clip,
            },
            "sequence": out_seq,
            "note": "sequence item = [x,y,dt,p,pen] or [x,y,dt,pen]"
        }
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("saved:", args.out_path)

    if __name__ == "__main__":
        main()
