import os
# print (os.getcwd())
import torch.nn as nn
import copy
import torch
import numpy as np
import pickle
import torch.nn.functional as F


def get_transition_matrices(model, num_cls, eval_loader, device):
	model.eval()
	p = []
	T_spadesuit = np.zeros((num_cls,num_cls))
	all_labels = []
	all_indexs = []
	all_features = []
	with torch.no_grad():
		for batch_idx, (data, label, indexs, _) in enumerate(eval_loader):
			data = data.to(device)
			all_labels.append(label)
			all_indexs.append(indexs)
			label = label.to(device)
			feat, output = model(data)
			all_features.append(feat.detach().cpu().numpy())
			probs = F.softmax(output, dim=1).cpu().data.numpy()
			_, output = output.topk(1, 1, True, True)           
			output = output.view(-1).cpu().data
			label = label.view(-1).cpu().data
			for i in range(len(label)): 
				T_spadesuit[int(output[i])][int(label[i])]+=1
			p += probs[:].tolist()  
	T_spadesuit = np.array(T_spadesuit)
	sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(num_cls,1)).transpose()
	T_spadesuit = T_spadesuit/sum_matrix
	p = np.array(p)
	T_clubsuit = est_t_matrix(p,filter_outlier=True)
	T_spadesuit = np.nan_to_num(T_spadesuit)
	all_labels = np.hstack(all_labels)
	all_indexs = np.hstack(all_indexs)
	all_features = np.concatenate(all_features, axis=0)
	return T_spadesuit, T_clubsuit, all_labels, all_indexs, all_features

def est_t_matrix(eta_corr, filter_outlier=False):

	# number of classes
	num_classes = eta_corr.shape[1]
	T = np.empty((num_classes, num_classes))

	# find a 'perfect example' for each class
	for i in np.arange(num_classes):

		if not filter_outlier:
			idx_best = np.argmax(eta_corr[:, i])
		else:
			eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
			robust_eta = eta_corr[:, i]
			robust_eta[robust_eta >= eta_thresh] = 0.0
			idx_best = np.argmax(robust_eta)

		for j in np.arange(num_classes):
			T[i, j] = eta_corr[idx_best, j]
	return T


				