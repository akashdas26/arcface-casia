"""Final evaluation on the held-out TEST identities (run once, after training)."""
import argparse
import torch
from config import Config
from datasets.split import get_pairs, make_split
from datasets.verification import evaluate
from models.iresnet import build_backbone


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{cfg.checkpoint_dir}/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = make_split(cfg)
    ck = torch.load(args.ckpt, map_location=device)
    model = build_backbone(cfg.backbone, cfg.embedding_size, cfg.dropout).to(device)
    model.load_state_dict(ck["backbone"])

    # Threshold chosen on VAL, applied unchanged to TEST (no tuning on test data).
    thr = ck["threshold"]
    r = evaluate(model, cfg.train_dir, get_pairs(cfg, split, "test"), device, threshold=thr,
                 num_workers=cfg.num_workers)
    print(f"TEST verification accuracy: {r['acc']:.4f}  (threshold {thr:.3f} from val, "
          f"best val acc {ck['val_acc']:.4f})")


if __name__ == "__main__":
    main()
