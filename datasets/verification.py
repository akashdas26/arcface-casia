"""Pair-verification evaluation on held-out identities."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from .casia import ImageListDataset, eval_transform


@torch.no_grad()
def extract(model, root, paths, device, batch_size=256, num_workers=4):
    loader = DataLoader(ImageListDataset(root, paths, eval_transform()),
                        batch_size=batch_size, num_workers=num_workers)
    model.eval()
    feats = []
    for x in loader:
        x = x.to(device)
        f = model(x) + model(torch.flip(x, dims=[3]))     # flip-TTA
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(feats).numpy()


def _acc_at(th, sims, labels):
    return ((sims > th) == labels).mean()


def best_threshold(sims, labels):
    cands = np.linspace(-1, 1, 2001)
    accs = [_acc_at(t, sims, labels) for t in cands]
    return float(cands[int(np.argmax(accs))]), float(max(accs))


def pair_similarities(model, root, pairs, device, **kw):
    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    idx = {p: i for i, p in enumerate(uniq)}
    feats = extract(model, root, uniq, device, **kw)
    sims = np.array([float(feats[idx[a]] @ feats[idx[b]]) for a, b, _ in pairs])
    labels = np.array([bool(l) for _, _, l in pairs])
    return sims, labels


def evaluate(model, root, pairs, device, threshold=None, **kw):
    """If threshold is None it is chosen on these pairs (use for VAL only).
    For TEST, pass the threshold found on VAL."""
    sims, labels = pair_similarities(model, root, pairs, device, **kw)
    if threshold is None:
        threshold, acc = best_threshold(sims, labels)
    else:
        acc = _acc_at(threshold, sims, labels)
    return {"acc": float(acc), "threshold": float(threshold)}
