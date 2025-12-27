\
    from __future__ import annotations
    import os, time, json, argparse
    from typing import Any, Dict

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset import StrokeBCDataset, collate_batch
    from model import GRUPolicy
    from utils import set_seed, device_from_arg, ensure_dir, RunningMean, save_jsonl, save_checkpoint

    def load_yaml_or_default(path: str | None) -> Dict[str, Any]:
        if path is None:
            return {}
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--data_dir", required=True, help="원본 session json들이 있는 폴더 (예: data/raw)")
        ap.add_argument("--run_dir", required=True, help="출력 폴더 (예: runs/exp1)")
        ap.add_argument("--config", default=None, help="config.yaml 경로(선택)")
        ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        ap.add_argument("--seed", type=int, default=None)

        ap.add_argument("--epochs", type=int, default=20)
        ap.add_argument("--batch_size", type=int, default=32)
        ap.add_argument("--seq_len", type=int, default=128)
        ap.add_argument("--lr", type=float, default=1e-3)
        ap.add_argument("--weight_decay", type=float, default=0.0)
        ap.add_argument("--grad_clip", type=float, default=1.0)
        ap.add_argument("--num_workers", type=int, default=0)

        ap.add_argument("--normalize", type=int, default=1)
        ap.add_argument("--use_pressure", type=int, default=1)
        ap.add_argument("--dt_clip", type=float, default=0.5)

        ap.add_argument("--hidden", type=int, default=256)
        ap.add_argument("--layers", type=int, default=2)
        ap.add_argument("--dropout", type=float, default=0.1)
        ap.add_argument("--max_files", type=int, default=None)

        args = ap.parse_args()

        cfg = load_yaml_or_default(args.config)
        ensure_dir(args.run_dir)
        ensure_dir(os.path.join(args.run_dir, "checkpoints"))

        # config 기록
        cfg_path = os.path.join(args.run_dir, "config_used.yaml")
        try:
            import yaml
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"yaml": cfg, "cli": vars(args)}, f, allow_unicode=True, sort_keys=False)
        except Exception:
            with open(os.path.join(args.run_dir, "config_used.json"), "w", encoding="utf-8") as f:
                json.dump({"yaml": cfg, "cli": vars(args)}, f, ensure_ascii=False, indent=2)

        set_seed(args.seed)
        device = device_from_arg(args.device)

        ds = StrokeBCDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            normalize=bool(args.normalize),
            use_pressure=bool(args.use_pressure),
            dt_clip=args.dt_clip,
            max_files=args.max_files,
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=(device.type == "cuda"),
        )

        model = GRUPolicy(
            input_dim=ds.input_dim,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            output_dim=ds.output_dim,
        ).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        log_path = os.path.join(args.run_dir, "train_log.jsonl")
        if os.path.exists(log_path):
            os.remove(log_path)

        best_loss = float("inf")
        step = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            rm = RunningMean()
            pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                if ds.output_dim == 5:
                    reg_pred = pred[..., 0:4]
                    pen_logit = pred[..., 4:5]
                    reg_tgt = yb[..., 0:4]
                    pen_tgt = yb[..., 4:5]
                else:
                    reg_pred = pred[..., 0:3]
                    pen_logit = pred[..., 3:4]
                    reg_tgt = yb[..., 0:3]
                    pen_tgt = yb[..., 3:4]

                loss_reg = mse(reg_pred, reg_tgt)
                loss_pen = bce(pen_logit, pen_tgt)
                loss = loss_reg + 0.2 * loss_pen

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                step += 1
                rm.update(loss.item(), n=xb.size(0))
                pbar.set_postfix(loss=rm.value)

            epoch_loss = rm.value
            save_jsonl(log_path, {"epoch": epoch, "loss": epoch_loss, "time": time.time()})

            # checkpoints
            last_ckpt = os.path.join(args.run_dir, "checkpoints", "last.pt")
            save_checkpoint(last_ckpt, model, optim, step, best_loss, {"yaml": cfg, "cli": vars(args)})
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_ckpt = os.path.join(args.run_dir, "checkpoints", "best.pt")
                save_checkpoint(best_ckpt, model, optim, step, best_loss, {"yaml": cfg, "cli": vars(args)})

            print(f"[epoch {epoch}] loss={epoch_loss:.6f} best={best_loss:.6f}")

        print("done.")
        print("best checkpoint:", os.path.join(args.run_dir, "checkpoints", "best.pt"))

    if __name__ == "__main__":
        main()
