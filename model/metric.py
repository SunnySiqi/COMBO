import torch
import numpy as np

def metric_overall(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		correct = 0
		correct += torch.sum(pred == target).item()
	return correct / len(target)

def metric_noisy(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		correct = 0
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		idxs = np.where((target < 5) == True)[0]
		correct += np.sum(pred[idxs] == target[idxs])
	return correct / len(idxs)

def metric_control(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		correct = 0
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		idxs = np.where((target >= 5) == True)[0]
		correct += np.sum(pred[idxs] == target[idxs])
	return correct / len(idxs)

def metric_top5(output, target, k=5):
	with torch.no_grad():
		pred = torch.topk(output, k, dim=1)[1]
		assert pred.shape[0] == len(target)
		correct = 0
		for i in range(k):
			correct += torch.sum(pred[:, i] == target).item()
	return correct / len(target)

def metric_cleancls(output, target, clean_classes):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		correct = 0
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		idxs = []
		for clean_c in clean_classes:
			idxs += list(np.where((target == clean_c) == True)[0])
		correct += np.sum(pred[idxs] == target[idxs])
	return correct / len(idxs)

def metric_weak(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		c_ids = np.where((target == 0) == True)[0]
		acc = np.sum(pred[c_ids] == target[c_ids])/len(c_ids)
	return acc

def metric_medium(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		c_ids = np.where((target == 1) == True)[0]
		acc = np.sum(pred[c_ids] == target[c_ids])/len(c_ids)
	return acc

def metric_strong(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		c_ids = np.where((target == 2) == True)[0]
		acc = np.sum(pred[c_ids] == target[c_ids])/len(c_ids)
	return acc

def metric_ctl(output, target):
	with torch.no_grad():
		pred = torch.argmax(output, dim=1)
		assert pred.shape[0] == len(target)
		pred = pred.cpu().detach().numpy()
		target = target.cpu().detach().numpy()
		c_ids = np.where((target == 3) == True)[0]
		acc = np.sum(pred[c_ids] == target[c_ids])/len(c_ids)
	return acc

def metric_per_class(output, target):
	pred = np.argmax(output, axis=1)
	acc = [0]*4
	for i in range(4):
		c_ids = np.where((target == i) == True)[0]
		acc[i] = np.sum(pred[c_ids] == target[c_ids])/len(c_ids)
	return acc


