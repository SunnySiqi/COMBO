from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix
import sys
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import copy

def fix_seed(seed=888):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def train_val_split(class_ind, dataset, seed):
	fix_seed(seed)
	if dataset == 'cifar10':
		num_classes = 10
	else:
		num_classes = 100
	train_n = int(50000 * 0.9 / num_classes)
	train_idxs = []
	val_idxs = []

	for i in range(num_classes):
		idxs = class_ind[i]
		np.random.shuffle(idxs)
		train_idxs.extend(idxs[:train_n])
		val_idxs.extend(idxs[train_n:])
	return train_idxs, val_idxs

def train_val_split_ctl(class_ind, dataset, non_ctl_sample, ctl_sample, seed):
	fix_seed(seed)
	if dataset == 'cifar10':
		num_classes = 10
		ctl_cls_lower_bound = 5
	else:
		num_classes = 100
		ctl_cls_lower_bound = 50

	non_ctl_train_n = int(non_ctl_sample*0.9)
	ctl_train_n = int(ctl_sample*0.9)
	non_ctl_val_n = int(non_ctl_sample*0.1)
	ctl_val_n = int(ctl_sample*0.1)
	train_idxs = []
	val_idxs = []

	for i in range(ctl_cls_lower_bound):
		idxs = class_ind[i]
		np.random.shuffle(idxs)
		train_idxs.extend(idxs[:non_ctl_train_n])
		left_idx = idxs[non_ctl_train_n:]
		np.random.shuffle(left_idx)
		val_idxs.extend(left_idx[:non_ctl_val_n])

	for i in range(ctl_cls_lower_bound, num_classes):
		idxs = class_ind[i]
		np.random.shuffle(idxs)
		train_idxs.extend(idxs[:ctl_train_n])
		left_idx = idxs[ctl_train_n:]
		np.random.shuffle(left_idx)
		val_idxs.extend(left_idx[:ctl_val_n])
	return train_idxs, val_idxs
		

def unpickle(file):
	import _pickle as cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo, encoding='latin1')
	return dict


class cifar_dataset(Dataset): 
	def __init__(self, dataset, mode, r, noise_mode, transform, noise_file=''): 
		
		self.r = r # noise ratio
		self.transform = transform
		self.mode = mode

		if dataset == 'cifar10':
			root_dir = '/net/ivcfs5/mnt/data/swang/research/baseline_fine/FINE_official-master/unicon/data/cifar10/cifar-10-batches-py'
			#root_dir = '/research/swang/research/baseline_fine/FINE_official-master/unicon/data/cifar10/cifar-10-batches-py'              
			num_class =10         
		else:
			root_dir = '/net/ivcfs5/mnt/data/swang/research/baseline_fine/FINE_official-master/unicon/data/cifar100/cifar-100-python'
			#root_dir = '/research/swang/research/baseline_fine/FINE_official-master/unicon/data/cifar100/cifar-100-python'
			num_class =100

		if noise_mode == 'asym':
			num_sample = 45000
		else:
			num_sample = 22500

		if self.mode=='test':
			if dataset=='cifar10':    
				test_dic = unpickle('%s/test_batch'%root_dir)
				self.test_data = test_dic['data']
				self.test_data = self.test_data.reshape((10000, 3, 32, 32))
				self.test_data = self.test_data.transpose((0, 2, 3, 1))  
				self.test_label = test_dic['labels']
			elif dataset=='cifar100':
				test_dic = unpickle('%s/test'%root_dir)
				self.test_data = test_dic['data']
				self.test_data = self.test_data.reshape((10000, 3, 32, 32))
				self.test_data = self.test_data.transpose((0, 2, 3, 1))  
				self.test_label = test_dic['fine_labels']                            
		
		else:    
			train_data=[]
			train_label=[]
			if dataset=='cifar10': 
				for n in range(1,6):
					dpath = '%s/data_batch_%d'%(root_dir,n)
					data_dic = unpickle(dpath)
					train_data.append(data_dic['data'])
					train_label = train_label+data_dic['labels']
				train_data = np.concatenate(train_data)
			elif dataset=='cifar100':    
				train_dic = unpickle('%s/train'%root_dir)
				train_data = train_dic['data']
				train_label = train_dic['fine_labels']
			train_data = train_data.reshape((50000, 3, 32, 32))
			train_data = train_data.transpose((0, 2, 3, 1))
			
			class_ind = {}
			for kk in range(num_class):
				class_ind[kk] = [i for i in range(len(train_data)) if train_label[i]==kk]

			# train val split
			if noise_mode == 'asym':
				train_idxs, val_idxs = train_val_split(class_ind, dataset, seed=47)
			else:
				if self.r == 0.2:
					non_ctl_sample = 2000
					ctl_sample = 3000
				elif self.r == 0.4:
					non_ctl_sample = 1800
					ctl_sample = 4200
				elif self.r == 0.5:
					non_ctl_sample = 1250
					ctl_sample = 3750
				elif self.r == 0.8:
					non_ctl_sample = 500
					ctl_sample = 4500
				if dataset == 'cifar100':
					non_ctl_sample = int(non_ctl_sample/10)
					ctl_sample = int(ctl_sample/10)
				train_idxs, val_idxs = train_val_split_ctl(class_ind, dataset, non_ctl_sample, ctl_sample, seed=47)

			if os.path.exists(noise_file):             
				noise_label = np.load(noise_file)['label']
			else:
				noise_label = train_label.copy()
				if noise_mode == 'asym':
					if dataset == 'cifar10':
						for i in range(num_class):
							indices = class_ind[i]
							np.random.shuffle(indices)
							for j, idx in enumerate(indices):
								if j < self.r * len(indices):
								# truck -> automobile
									if i == 9:
										noise_label[idx] = 1
									elif i == 1:
										noise_label[idx] = 9
								# cat -> dog
									elif i == 3:
										noise_label[idx] = 5
									elif i == 5:
										noise_label[idx] = 3
								# deer -> horse
									elif i == 4:
										noise_label[idx] = 7
									elif i == 7:
										noise_label[idx] = 4
					else:
						for i in range(num_class):
							indices = class_ind[i]
							np.random.shuffle(indices)
							for j, idx in enumerate(indices):
								if j < self.r * len(indices):
									# beaver (4) -> otter (55)
									# aquarium fish (1) -> flatfish (32)
									# poppies (62)-> roses (70)
									# bottles (9) -> cans (16)
									# apples (0) -> pears (57)
									# chair (20) -> couch (25)
									# bee (6) -> beetle (7)
									# lion (43)-> tiger (88)
									# crab (26) -> spider (79)
									# rabbit (65) -> squirrel (80)
									# maple (47) -> oak (52)
									# bicycle (8)-> motorcycle (48)
									if i == 4:
										noise_label[idx] = 55
									elif i == 55:
										noise_label[idx] = 4
									elif i == 1:
										noise_label[idx] = 32
									elif i == 32:
										noise_label[idx] = 1
									elif i == 62:
										noise_label[idx] = 70
									elif i == 70:
										noise_label[idx] = 62
									elif i == 9:
										noise_label[idx] = 16
									elif i == 16:
										noise_label[idx] = 9
									elif i == 0:
										noise_label[idx] = 57
									elif i == 57:
										noise_label[idx] = 0
									elif i == 20:
										noise_label[idx] = 25
									elif i == 25:
										noise_label[idx] = 20
									elif i == 6:
										noise_label[idx] = 7
									elif i == 7:
										noise_label[idx] = 6
									elif i == 43:
										noise_label[idx] = 88
									elif i == 88:
										noise_label[idx] = 43
									elif i == 26:
										noise_label[idx] = 79
									elif i == 79:
										noise_label[idx] = 26
									elif i == 65:
										noise_label[idx] = 80
									elif i == 80:
										noise_label[idx] = 65
									elif i == 47:
										noise_label[idx] = 52
									elif i == 52:
										noise_label[idx] = 47
									elif i == 8:
										noise_label[idx] = 48
									elif i == 48:
										noise_label[idx] = 8
				else:
					if dataset == 'cifar10':
						control_lower_idx = 5
						num_ctl_cls = 5
					else:
						control_lower_idx = 50
						num_ctl_cls = 50
					num_mislabel_per_ctl = int((0.9*ctl_sample - 0.9*non_ctl_sample)/2)
				
					for i in range(control_lower_idx, num_class):
						idxs = class_ind[i]
						np.random.shuffle(idxs)
						for j in range(num_mislabel_per_ctl):
							noise_label[idxs[j]] = np.random.randint(0, control_lower_idx)
				print("Save noisy labels to %s ..."%noise_file)        
				np.savez(noise_file, label = noise_label)    

			if self.mode=='val':
				self.val_data = train_data[val_idxs]
				self.val_labels_gt = np.array(train_label)[val_idxs]
				self.noise_label = np.array(noise_label)[val_idxs]
			else:
				self.train_data = train_data[train_idxs]
				self.train_labels_gt = np.array(train_label)[train_idxs]
				self.noise_label = np.array(noise_label)[train_idxs]
				self.whole_train_data = self.train_data.copy()
				self.whole_train_labels_gt = self.train_labels_gt.copy()
				self.whole_noise_label = self.noise_label.copy()

	def split_train(self, selected_idxs=[], cluster_labels=[], supervised=True):
		self.train_data = self.whole_train_data[selected_idxs]
		self.train_labels_gt = self.whole_train_labels_gt[selected_idxs]
		if supervised:
			self.noise_label = self.whole_noise_label[selected_idxs]
		else:
			self.noise_label = cluster_labels[selected_idxs]

	def distill_train(self, selected_idxs=[], bayes_labels=[]):
		self.train_data = self.whole_train_data[selected_idxs]
		self.train_labels_gt = self.noise_label
		self.noise_label = bayes_labels

	def __getitem__(self, index):
		if self.mode=='labeled':
			img, target = self.train_data[index], self.noise_label[index]
			image = Image.fromarray(img)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)

			return img1, img2, img3, img4, target 

		elif self.mode=='unlabeled':
			img, target = self.train_data[index], self.noise_label[index]
			image = Image.fromarray(img)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)
			return img1, img2, img3, img4, target

		elif self.mode=='all':
			img, target, gt = self.train_data[index], self.noise_label[index], self.train_labels_gt[index]
			img = Image.fromarray(img)
			img = self.transform(img)            
			return img, target, index, gt

		elif self.mode=='test':
			img, target = self.test_data[index], self.test_label[index]
			img = Image.fromarray(img)
			img = self.transform(img)            
			return img, target

		elif self.mode=='val':
			img, target = self.val_data[index], self.noise_label[index]
			img = Image.fromarray(img)
			img = self.transform(img)            
			return img, target
		   
	def __len__(self):
		if self.mode == 'test':
			return len(self.test_data)
		elif self.mode == 'val':
			return len(self.val_data)
		else:
			return len(self.train_data)   
		
class cifar_dataloader():  
	def __init__(self, dataset, r, noise_mode, batch_size, num_workers, noise_file=''):
		self.dataset = dataset
		self.r = r
		self.noise_mode = noise_mode
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.noise_file = noise_file
		
		if self.dataset=='cifar10':
			transform_weak_10 = transforms.Compose(
				[
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
				]
			)

			transform_strong_10 = transforms.Compose(
				[
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.AutoAugment(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
				]
			)

			self.transforms = {
				"warmup": transform_weak_10,
				"unlabeled": [
							transform_weak_10,
							transform_weak_10,
							transform_strong_10,
							transform_strong_10
						],
				"labeled": [
							transform_weak_10,
							transform_weak_10,
							transform_strong_10,
							transform_strong_10
						],
			}

			self.transform_test = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
				])

		elif self.dataset=='cifar100':
			transform_weak_100 = transforms.Compose(
				[
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
				]
			)

			transform_strong_100 = transforms.Compose(
				[
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.AutoAugment(),
					transforms.ToTensor(),
					transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
				]
			)

			self.transforms = {
				"warmup": transform_weak_100,
				"unlabeled": [
							transform_weak_100,
							transform_weak_100,
							transform_strong_100,
							transform_strong_100
						],
				"labeled": [
							transform_weak_100,
							transform_weak_100,
							transform_strong_100,
							transform_strong_100
						],
			}        
			self.transform_test = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
				])
				   
	def run(self, mode, supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]):
		if mode=='warmup':
			all_dataset = cifar_dataset(dataset=self.dataset, mode="all", r=self.r, noise_mode=self.noise_mode, transform=self.transforms["warmup"], noise_file=self.noise_file)                
			trainloader = DataLoader(
				dataset=all_dataset, 
				batch_size=self.batch_size*2,
				shuffle=True,
				num_workers=self.num_workers)             
			return trainloader
									 
		elif mode=='train':
			labeled_dataset = cifar_dataset(dataset=self.dataset, mode="labeled", r=self.r, noise_mode=self.noise_mode, transform=self.transforms["labeled"], noise_file=self.noise_file)  
			labeled_dataset.split_train(selected_idxs=supervised_idxs, cluster_labels=cluster_labels, supervised=True)           
			labeled_trainloader = DataLoader(
				dataset=labeled_dataset, 
				batch_size=self.batch_size,
				shuffle=True,
				num_workers=self.num_workers, drop_last=True)  
			unlabeled_dataset = cifar_dataset(dataset=self.dataset, mode="unlabeled", r=self.r, noise_mode=self.noise_mode, transform=self.transforms["unlabeled"], noise_file=self.noise_file)  
			if cluster_labels is not None:
				unlabeled_dataset.split_train(selected_idxs=semi_supervised_idxs, cluster_labels=cluster_labels, supervised=False)
			else:
				unlabeled_dataset.split_train(selected_idxs=semi_supervised_idxs, cluster_labels=cluster_labels, supervised=True)                         
			unlabeled_trainloader = DataLoader(
				dataset=unlabeled_dataset, 
				batch_size= self.batch_size,
				shuffle=True,
				num_workers=self.num_workers, drop_last =True)    

			return labeled_trainloader, unlabeled_trainloader                
		
		elif mode=='val':
			val_dataset = cifar_dataset(dataset=self.dataset, mode="val", r=self.r, noise_mode=self.noise_mode, transform=self.transform_test, noise_file=self.noise_file)       
			val_loader = DataLoader(
				dataset=val_dataset, 
				batch_size=100,
				shuffle=False,
				num_workers=self.num_workers)          
			return val_loader

		elif mode=='test':
			test_dataset = cifar_dataset(dataset=self.dataset, mode="test", r=self.r, noise_mode=self.noise_mode, transform=self.transform_test, noise_file=self.noise_file)       
			test_loader = DataLoader(
				dataset=test_dataset, 
				batch_size=100,
				shuffle=False,
				num_workers=self.num_workers)          
			return test_loader
		
		elif mode=='distill':
			distill_dataset = cifar_dataset(dataset=self.dataset, mode="all", r=self.r, noise_mode=self.noise_mode, transform=self.transform_test, noise_file=self.noise_file)      
			distill_dataset.distill_train(selected_idxs=supervised_idxs, bayes_labels=cluster_labels)
			distill_loader = DataLoader(
				dataset=distill_dataset, 
				batch_size=100,
				shuffle=False,
				num_workers=self.num_workers, drop_last= True)    
			return distill_loader
		
		elif mode=='eval_train':
			eval_dataset = cifar_dataset(dataset=self.dataset, mode="all", r=self.r, noise_mode=self.noise_mode, transform=self.transform_test, noise_file=self.noise_file)      
			eval_loader = DataLoader(
				dataset=eval_dataset, 
				batch_size=100,
				shuffle=False,
				num_workers=self.num_workers, drop_last= True)          
			return eval_loader 
