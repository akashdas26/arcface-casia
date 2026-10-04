import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


class FaceDataset(Dataset):
    """Training set built from the `train` part of the identity split."""

    def __init__(self, root, id2imgs, transform):
        self.root, self.transform = root, transform
        self.classes = sorted(id2imgs)
        self.samples = [(p, label) for label, n in enumerate(self.classes) for p in id2imgs[n]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(os.path.join(self.root, p)).convert("RGB")
        return self.transform(img), y


class ImageListDataset(Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        return self.transform(img)
