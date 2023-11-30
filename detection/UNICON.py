import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import numpy as np
from tqdm import tqdm
# from memory_profiler import profile


# KL divergence
# with k: dim=0; without k: dim=1
def kl_divergence(p, q, with_knowledge=False):
	if with_knowledge:
		return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=0)
	else:
		return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
	def __init__(self):
		super(Jensen_Shannon,self).__init__()
		pass
	def forward(self, p,q, with_knowledge=False):
		m = (p+q)/2
		return 0.5*kl_divergence(p, m, with_knowledge) + 0.5*kl_divergence(q, m, with_knowledge)

## Calculate JSD
def Calculate_JSD(model1, num_samples, eval_loader, num_cls, model2=None):  
	JS_dist = Jensen_Shannon()
	JSD   = torch.zeros(num_samples)    

	for batch_idx, (inputs, targets, index, _) in enumerate(eval_loader):
		inputs, targets = inputs.cuda(), targets.cuda()
		batch_size = inputs.size()[0]

		## Get outputs of both network
		with torch.no_grad():
			out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])
			
			if model2:    
				out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])
				out = (out1 + out2)/2
			else:
				out = out1    

		## Divergence clculator to record the diff. between ground truth and output prob. dist.  
		dist = JS_dist(out,  F.one_hot(targets, num_classes = num_cls), with_knowledge=False)
		JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

	return JSD

def Calculate_JSD_K(model1, num_samples, eval_loader, noise_source_dict, clean_classes, num_cls, model2=None):  
	JS_dist = Jensen_Shannon()
	JSD   = torch.zeros(num_samples)
	# noise_source_dict = {}
	# for p in args.noise_source:
	#     noise_source_dict[int(p[0])] = np.array([int(p[1])])
	with tqdm(eval_loader) as progress:
		for batch_idx, (inputs, targets, index, _) in enumerate(progress):
			inputs, targets = inputs.cuda(), targets.cuda()
			batch_size = inputs.size()[0]

			## Get outputs of both network
			with torch.no_grad():
				out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
				if model2:    
					out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])
					out = (out1 + out2)/2
				else:
					out = out1    

			## Divergence clculator to record the diff. between ground truth and output prob. dist.  
			for i in range(batch_size):
				label = targets[i].item()
				if clean_classes and label in clean_classes:
					JSD[int(batch_idx*batch_size)+i] = 0
					continue
				label_dist = JS_dist(out[i],  F.one_hot(targets[i], num_classes = num_cls), with_knowledge=True)
				if label not in noise_source_dict:
					JSD[int(batch_idx*batch_size)+i] = label_dist
					continue
				min_source_dist = min([JS_dist(out[i],  F.one_hot(torch.tensor(source).to(out.device), num_classes = num_cls), with_knowledge=True) for source in noise_source_dict[label]])
				if label_dist < min_source_dist:
					JSD[int(batch_idx*batch_size)+i] = label_dist
				else:
					JSD[int(batch_idx*batch_size)+i] = 1
	return JSD

def select_idx_UNICON(model1, eval_loader, noise_source_dict, clean_classes, d_u, tau, num_cls, model2=None):
	num_samples = eval_loader.dataset.__len__()
	prob = Calculate_JSD_K(model1, num_samples, eval_loader, noise_source_dict, clean_classes, num_cls, model2)       
	threshold = torch.sum(prob)/torch.count_nonzero(prob)
	if threshold.item()>d_u:
		threshold = threshold - (threshold-torch.min(prob))/tau
	select_idx = np.where(prob.detach().cpu().numpy() < threshold.detach().cpu().numpy())
	return select_idx

	               

	