from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os

class animal10N_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_class=10): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            
        
        if mode == 'test':
            self.test_imgs = []
            test_files=os.listdir('%s/testing'%self.root)
            for i in range(len(test_files)):
                label = int(test_files[i].split('_')[0][0])
                self.test_labels[test_files[i]] = label
                self.test_imgs.append(test_files[i])   
        
        elif mode == 'val':
            self.val_imgs = []
            test_files=os.listdir('%s/testing'%self.root)
            for i in range(len(test_files)):
                label = int(test_files[i].split('_')[0][0])
                self.val_labels[test_files[i]] = label
                self.val_imgs.append(test_files[i]) 
        else:
            self.train_imgs = []
            train_files=os.listdir('%s/training'%self.root)
            for i in range(len(train_files)):
                label = int(train_files[i].split('_')[0][0])
                self.train_labels[train_files[i]] = label
                self.train_imgs.append(train_files[i])
            random.shuffle(self.train_imgs)
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
            img_path = '%s/training/'%self.root + self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target    
        
        elif self.mode == "unlabeled":
            img_path = '%s/training/'%self.root + self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target
            
        elif self.mode == "all":
            img_path = '%s/training/'%self.root + self.train_imgs[index]
            target = self.train_labels[self.train_imgs[index]]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, index, index

        elif self.mode == "test":
            img_path = '%s/testing/'%self.root + self.test_imgs[index]
            target = self.test_labels[self.test_imgs[index]]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target

        elif self.mode=='val':
            img_path = '%s/testing/'%self.root + self.val_imgs[index]
            target = self.val_labels[self.val_imgs[index]]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
                           

class animal10N_dataloader:
    def __init__(
        self,
        root,
        batch_size,
        warmup_batch_size,
        num_workers    ):
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root = root

        animal10N_weak_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )


        animal10N_strong_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.transforms = {
            "warmup": animal10N_weak_transform,
            "unlabeled": [
                        animal10N_strong_transform,
                        animal10N_strong_transform,
                        animal10N_weak_transform,
                        animal10N_weak_transform
                    ],
            "labeled": [
                        animal10N_strong_transform,
                        animal10N_strong_transform,
                        animal10N_weak_transform,
                        animal10N_weak_transform
                    ]
        }
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def run(self, mode, supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]):
        if mode == "warmup":
            warmup_dataset = animal10N_dataset(
                self.root,
                transform=self.transforms["warmup"],
                mode="all"
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = animal10N_dataset(
                self.root, 
                transform=self.transforms["labeled"],
                mode="labeled"
            )
            labeled_dataset.split_train(selected_idxs=supervised_idxs, cluster_labels=cluster_labels, supervised=True)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True
            )
            unlabeled_dataset = animal10N_dataset(
                self.root, 
                transform = self.transforms["unlabeled"],
                mode = "unlabeled"
            )
            if cluster_labels is not None:
                unlabeled_dataset.split_train(selected_idxs=semi_supervised_idxs, cluster_labels=cluster_labels, supervised=False)
            else:
                unlabeled_dataset.split_train(selected_idxs=semi_supervised_idxs, cluster_labels=cluster_labels, supervised=True)                         
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size= self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last =True)    

            return labeled_loader, unlabeled_trainloader

        elif mode == "eval_train":
            eval_dataset = animal10N_dataset(
                self.root, 
                transform=self.transforms_test,
                mode="all"
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = animal10N_dataset(
                self.root,  transform=self.transforms_test, mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
        elif mode=='distill':
            distill_dataset = animal10N_dataset(
                self.root,  transform=self.transforms_test, mode="all"
            )
            distill_dataset.distill_train(selected_idxs=supervised_idxs, bayes_labels=cluster_labels)
            distill_loader = DataLoader(
                dataset=distill_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last=True)    
            return distill_loader

        elif mode == "val":
            val_dataset = animal10N_dataset(
                self.root ,transform=self.transforms_test, mode="val"
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader

    
        