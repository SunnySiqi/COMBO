from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
import random


class clothing1M_dataset(Dataset):
    def __init__(self, root, transform, mode, num_class=14, num_imgs: int=256000):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.noisy_labels = {}
        self.clean_labels = {}
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open("%s/annotations/noisy_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.noisy_labels[img_path] = int(entry[1])
        with open("%s/annotations/clean_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.clean_labels[img_path] = int(entry[1])

        if mode == "test":
            self.test_imgs = []
            with open("%s/annotations/clean_test_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    self.test_imgs.append(img_path)
            for test_img in self.test_imgs:
                self.test_labels[test_img] = self.clean_labels[test_img]

        elif mode == "val":
            self.val_imgs = []
            with open("%s/annotations/clean_val_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    self.val_imgs.append(img_path)
            # self.val_imgs = self.val_imgs[:256]  ## TODO remove this line
            for val_img in self.val_imgs:
                self.val_labels[val_img] = self.clean_labels[val_img]
        else:
            self.train_imgs = []
            with open("%s/annotations/noisy_train_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    self.train_imgs.append(img_path)
            for train_img in self.train_imgs:
                self.train_labels[train_img] = self.noisy_labels[train_img]
            random.shuffle(self.train_imgs)
            self.num_imgs = num_imgs
            self.train_imgs = self.train_imgs[:num_imgs]
            #     :300000
            # ]  # Use subset to train  ## TODO: set it back to 500000
            print("LEN OF THE TRAINING IMAGES!!!!!!!!", len(self.train_imgs))
            self.whole_train_imgs = np.array(self.train_imgs.copy())
            self.whole_train_labels = self.train_labels.copy()

    def split_train(self, selected_idxs=[], cluster_labels=[], supervised=True):
        self.train_imgs = self.whole_train_imgs[selected_idxs]
        self.train_labels = {}
        for i in range(len(self.train_imgs)):
            if supervised:
                self.train_labels[self.train_imgs[i]] = self.whole_train_labels[self.train_imgs[i]]
            else:
                self.train_labels[self.train_imgs[i]] = cluster_labels[selected_idxs[i]]

    def distill_train(self, selected_idxs=[], bayes_labels=[]):
        self.train_data = self.whole_train_imgs[selected_idxs]
        self.noise_label = bayes_labels

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            # image = Image.open(img_path).convert("RGB")
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img1 = self.transform[0](image) 
                img2 = self.transform[1](image)
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4, target

        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            # image = Image.open(img_path).convert("RGB")
            # img1 = self.transform[0](image)
            # img2 = self.transform[1](image)
            # img3 = self.transform[2](image)
            # img4 = self.transform[3](image)
            # return img1, img2, img3, img4, target
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img1 = self.transform[0](image) 
                img2 = self.transform[1](image)
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4, target

        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            # image = Image.open(img_path).convert("RGB")
            # img = self.transform(image)
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img = self.transform(image) 
            return img, target, index, index

        elif self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[self.test_imgs[index]]
            # image = Image.open(img_path).convert("RGB")
            # img = self.transform(image)
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img = self.transform(image) 
            return img, target

        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.val_labels[self.val_imgs[index]]
            # image = Image.open(img_path).convert("RGB")
            # img = self.transform(image)
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                img = self.transform(image) 
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing1M_dataloader:
    def __init__(self, root, batch_size, warmup_batch_size, num_workers, num_imgs:int):
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root = root
        self.num_imgs = num_imgs

        clothing1M_weak_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        clothing1M_strong_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        self.transforms = {
            "warmup": clothing1M_weak_transform,
            "unlabeled": [
                clothing1M_strong_transform,
                clothing1M_strong_transform,
                clothing1M_weak_transform,
                clothing1M_weak_transform,
            ],
            "labeled": [
                clothing1M_strong_transform,
                clothing1M_strong_transform,
                clothing1M_weak_transform,
                clothing1M_weak_transform,
            ],
        }
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

    def run(self, mode, supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]):
        if mode == "warmup":
            warmup_dataset = clothing1M_dataset(
                self.root, transform=self.transforms["warmup"], mode="all", num_imgs=self.num_imgs
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = clothing1M_dataset(
                self.root, transform=self.transforms["labeled"], mode="labeled", num_imgs=self.num_imgs
            )
            labeled_dataset.split_train(
                selected_idxs=supervised_idxs, cluster_labels=cluster_labels, supervised=True
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            unlabeled_dataset = clothing1M_dataset(
                self.root, transform=self.transforms["unlabeled"], mode="unlabeled", num_imgs=self.num_imgs
            )
            if cluster_labels is not None:
                unlabeled_dataset.split_train(
                    selected_idxs=semi_supervised_idxs,
                    cluster_labels=cluster_labels,
                    supervised=False,
                )
            else:
                unlabeled_dataset.split_train(
                    selected_idxs=semi_supervised_idxs,
                    cluster_labels=cluster_labels,
                    supervised=True,
                )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

            return labeled_loader, unlabeled_trainloader

        elif mode == "eval_train":
            eval_dataset = clothing1M_dataset(self.root, transform=self.transforms_test, mode="all", num_imgs=self.num_imgs)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = clothing1M_dataset(
                self.root, transform=self.transforms_test, mode="test", num_imgs=self.num_imgs
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
        elif mode == "distill":
            distill_dataset = clothing1M_dataset(
                self.root, transform=self.transforms_test, mode="all", num_imgs=self.num_imgs
            )
            distill_dataset.distill_train(
                selected_idxs=supervised_idxs, bayes_labels=cluster_labels
            )
            distill_loader = DataLoader(
                dataset=distill_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
            )
            return distill_loader

        elif mode == "val":
            val_dataset = clothing1M_dataset(self.root, transform=self.transforms_test, mode="val", num_imgs=self.num_imgs)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader
