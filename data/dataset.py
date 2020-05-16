import random
import os
from os.path import join
from glob import glob

import torch
from PIL import Image

from utils import get_image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dirs, transforms=None):
        self._transforms = transforms

        self._data = self._load_data(root_dirs)

    def __getitem__(self, idx):
        image = Image.open(self._data[idx]["path"])
        if self._transforms is not None:
            image = self._transforms(image)

        label = self._data[idx]["label"]
        return image, label

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def num_classes(self):
        return len(set([d["label"] for d in self._data]))

    @staticmethod
    def _load_data(root_dirs):
        data = []

        first_label = 0

        for root_dir in root_dirs:
            for label_id, label in enumerate(
                os.listdir(root_dir), start=first_label
            ):
                for path in sorted(
                    glob(join(root_dir, label, "*.jpg"))
                ):
                    d = {"path": path, "label": label_id}
                    data.append(d)

                first_label = data[-1]["label"] + 1

        print(
            "Data loaded. Data has {} images/labels ({} sets).".format(
                len(data), len(set([d["label"] for d in data]))
            )
        )

        return data


class TripletDataset(Dataset):
    def __getitem__(self, idx):
        anc = self._data[idx]

        pos_candidates = [d for d in self._data if d["label"] == anc["label"]]
        neg_candidates = [d for d in self._data if d["label"] != anc["label"]]

        pos = random.choice(
            [
                d
                for d in pos_candidates
                if d["path"] != anc["path"]
            ]
        )
        neg = random.choice(neg_candidates)

        anc_img = Image.open(anc["path"])
        pos_img = Image.open(pos["path"])
        neg_img = Image.open(neg["path"])

        if self._transforms is not None:
            anc_img = self._transforms(anc_img)
            pos_img = self._transforms(pos_img)
            neg_img = self._transforms(neg_img)

        return (anc_img, pos_img, neg_img), (anc, pos, neg)


class QuadrupletDataset(Dataset):
    def __getitem__(self, idx):
        anc = self._data[idx]

        pos_candidates = [d for d in self._data if d["label"] == anc["label"]]
        neg_candidates = [d for d in self._data if d["label"] != anc["label"]]

        ch = anc["ch"] if random.random() <= 0.3 else 3 - anc["ch"]
        pos = random.choice(
            [
                d
                for d in pos_candidates
                if d["path"] != anc["path"] and d["ch"] == ch
            ]
        )
        neg1 = random.choice([d for d in neg_candidates if d["ch"] == ch])
        neg2 = random.choice(
            [
                d
                for d in neg_candidates
                if d["label"] != neg1["label"] and d["ch"] == ch
            ]
        )

        anc_img = Image.open(anc["path"])
        pos_img = Image.open(pos["path"])
        neg1_img = Image.open(neg1["path"])
        neg2_img = Image.open(neg2["path"])

        if self._transforms is not None:
            anc_img = self._transforms(anc_img)
            pos_img = self._transforms(pos_img)
            neg1_img = self._transforms(neg1_img)
            neg2_img = self._transforms(neg2_img)

        return (anc_img, pos_img, neg1_img, neg2_img), (anc, pos, neg1, neg2)


class MOTDFDataset(torch.utils.data.Dataset):
    def __init__(self, mot_df, transforms=None):
        self._df = mot_df
        self._transforms = transforms

    def __getitem__(self, idx):
        series = self._df.iloc[idx]
        image = get_image(series, crop_bbox=True, pil=True)
        if self._transforms is not None:
            image = self._transforms(image)
        return image

    def __len__(self):
        return self._df.shape[0]
