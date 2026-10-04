"""Identity-disjoint train/val/test split + verification pair generation."""
import json
import os
import random

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def scan_identities(root, min_imgs=2):
    """Return {identity: [relative image paths]} for identities with >= min_imgs images."""
    ids = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        if len(imgs) >= min_imgs:
            ids[name] = [os.path.join(name, f) for f in imgs]
    return ids


def make_split(cfg):
    """Create (or load the cached) identity-disjoint split."""
    if os.path.exists(cfg.split_file):
        with open(cfg.split_file) as f:
            return json.load(f)

    ids = scan_identities(cfg.train_dir, cfg.min_imgs_per_id)
    names = sorted(ids)
    n_val = max(1, int(round(len(names) * cfg.val_id_ratio)))
    n_test = max(1, int(round(len(names) * cfg.test_id_ratio)))
    if len(names) < n_val + n_test + 2:
        raise ValueError(
            f"Only {len(names)} usable identities found in {cfg.train_dir}; need many more "
            f"for an identity-disjoint split. Point train_dir at the full CASIA-WebFace "
            f"(10,572 identities)."
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(names)
    split_ids = {
        "val": sorted(names[:n_val]),
        "test": sorted(names[n_val:n_val + n_test]),
        "train": sorted(names[n_val + n_test:]),
    }
    split = {k: {i: ids[i] for i in v} for k, v in split_ids.items()}

    os.makedirs(os.path.dirname(cfg.split_file) or ".", exist_ok=True)
    with open(cfg.split_file, "w") as f:
        json.dump(split, f)
    return split


def make_pairs(id2imgs, num_pairs, seed):
    """Balanced list of (path1, path2, is_same)."""
    rng = random.Random(seed)
    names = sorted(id2imgs)
    pairs = []
    for _ in range(num_pairs // 2):                       # positives
        n = rng.choice(names)
        a, b = rng.sample(id2imgs[n], 2)
        pairs.append((a, b, 1))
    for _ in range(num_pairs // 2):                       # negatives
        n1, n2 = rng.sample(names, 2)
        pairs.append((rng.choice(id2imgs[n1]), rng.choice(id2imgs[n2]), 0))
    rng.shuffle(pairs)
    return pairs


def get_pairs(cfg, split, which):
    path = os.path.join(cfg.pairs_dir, f"{which}_pairs.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    n = cfg.num_val_pairs if which == "val" else cfg.num_test_pairs
    pairs = make_pairs(split[which], n, cfg.seed + (1 if which == "val" else 2))
    os.makedirs(cfg.pairs_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f)
    return pairs
