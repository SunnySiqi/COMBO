from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import json
import torch
import os
import skimage.io
from .tps_transform import TPSTransform
#training: 36360 in total (0.9 train 32724, 0.1 val 3636)

class CP_dataset(Dataset): 
	def __init__(self, root, transform, mode, num_class=4): 
		
		self.root = root
		self.transform = transform
		self.mode = mode
		self.train_labels = {}
		self.test_labels = {}
		self.val_labels = {}

		self.classes_dict = {
				"BRD-A29260609": 0,
				"BRD-K04185004": 1,
				"BRD-K21680192": 2,
				"DMSO": 3,
		}
		
		self.metadata = pd.read_csv(self.root+'/cp.csv')
		if mode=='test':
			self.test_imgs = []
			for i in range(len(self.metadata)):
				if self.metadata.loc[i, "train_test_split"] == 'Task_one':
					self.test_imgs.append(self.metadata.loc[i, "file_path"])
					self.test_labels[self.metadata.loc[i, "file_path"]] = self.classes_dict[self.metadata.loc[i, "label"]]  
		else:
			train_imgs = []
			for i in range(len(self.metadata)):
				if self.metadata.loc[i, "train_test_split"] == 'Train':
					train_imgs.append(self.metadata.loc[i, "file_path"])
					self.train_labels[self.metadata.loc[i, "file_path"]] = self.classes_dict[self.metadata.loc[i, "label"]]
			num_train = len(train_imgs)
			train_val_split_idx = int(0.9*num_train)
			random.Random(47).shuffle(train_imgs)
			if mode == 'val':
				self.val_imgs = np.array(list(train_imgs)[train_val_split_idx:])
				self.val_labels = self.train_labels
			else:
				self.train_imgs = np.array(list(train_imgs)[:train_val_split_idx])
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
			img_path = '%s/images/'%self.root + self.train_imgs[index]
			target = self.train_labels[self.train_imgs[index]]
			img = skimage.io.imread(img_path)
			img = np.reshape(img, (img.shape[0], 160, -1), order="F")
			image = transforms.ToTensor()(img)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)
			return img1, img2, img3, img4, target     
		
		elif self.mode == "unlabeled":
			img_path = '%s/images/'%self.root + self.train_imgs[index]
			target = self.train_labels[self.train_imgs[index]]
			img = skimage.io.imread(img_path)
			img = np.reshape(img, (img.shape[0], 160, -1), order="F")
			image = transforms.ToTensor()(img)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)
			return img1, img2, img3, img4, target
			
		elif self.mode == "all":
			img_path = '%s/images/'%self.root + self.train_imgs[index]
			target = self.train_labels[self.train_imgs[index]]
			img = skimage.io.imread(img_path)
			img = np.reshape(img, (img.shape[0], 160, -1), order="F")
			image = transforms.ToTensor()(img)
			img = self.transform(image)
			return img, target, index, index

		elif self.mode == "test":
			img_path = '%s/images/'%self.root + self.test_imgs[index]
			target = self.test_labels[self.test_imgs[index]]
			img = skimage.io.imread(img_path)
			img = np.reshape(img, (img.shape[0], 160, -1), order="F")
			image = transforms.ToTensor()(img)
			img = self.transform(image)
			return img, target

		elif self.mode=='val':
			img_path = '%s/images/'%self.root + self.val_imgs[index]
			target = self.train_labels[self.val_imgs[index]]     
			img = skimage.io.imread(img_path)
			img = np.reshape(img, (img.shape[0], 160, -1), order="F")
			image = transforms.ToTensor()(img)   
			img = self.transform(image) 
			return img, target    
		
	def __len__(self):
		if self.mode=='test':
			return len(self.test_imgs)
		if self.mode=='val':
			return len(self.val_imgs)
		else:
			return len(self.train_imgs)            
		

class CP_dataloader:
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

		CP_weak_transform = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.RandomResizedCrop(
					224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
				transforms.RandomHorizontalFlip(),
				transforms.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
		(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
			]
		)


		CP_strong_transform = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.RandomResizedCrop(
					224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
				transforms.RandomHorizontalFlip(),
				TPSTransform(p=1),
				transforms.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
		(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
			]
		)

		self.transforms = {
			"warmup": CP_weak_transform,
			"unlabeled": [
						CP_strong_transform,
						CP_strong_transform,
						CP_weak_transform,
						CP_weak_transform
					],
			"labeled": [
						CP_strong_transform,
						CP_strong_transform,
						CP_weak_transform,
						CP_weak_transform
					]
		}
		self.transforms_test = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
		(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
			]
		)

	def run(self, mode, supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]):
		if mode == "warmup":
			warmup_dataset = CP_dataset(
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
			labeled_dataset = CP_dataset(
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
			unlabeled_dataset = CP_dataset(
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
			eval_dataset = CP_dataset(
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
			test_dataset = CP_dataset(
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
			distill_dataset = CP_dataset(
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
			val_dataset = CP_dataset(
				self.root ,transform=self.transforms_test, mode="val"
			)
			val_loader = DataLoader(
				dataset=val_dataset,
				batch_size=self.warmup_batch_size,
				shuffle=False,
				num_workers=self.num_workers,
			)
			return val_loader

		