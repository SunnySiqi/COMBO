import torch
import numpy as np
import pandas as pd
from sklearn import cluster
from tqdm import tqdm
from .gmm import *
from .util import *

__all__=['get_mean_vector', 'get_singular_vector', 'cleansing', 'fine', 'fine_w_noise_source', 'extract_cleanidx']


def get_mean_vector(mean_vector_dict, features, labels):
	with tqdm(total=len(np.unique(labels))) as pbar:
		for index in np.unique(labels):
			v = np.mean(features[labels==index], axis=0)
			mean_vector_dict[int(index)] = v
			pbar.update(1)
			
	return mean_vector_dict
			
def get_singular_vector(singular_vector_dict, features, labels):
	'''
	To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
	features: hidden feature vectors of data (numpy)
	labels: correspoding label list
	'''
	with tqdm(total=len(np.unique(labels))) as pbar:
		for index in np.unique(labels):
			_, _, v = np.linalg.svd(features[labels==index])
			singular_vector_dict[int(index)] = v[0]
			pbar.update(1)

	return singular_vector_dict

def get_score_w_noise_source(noise_source_dict, singular_vector_dict, features, labels, normalization):
	scores = []
	for indx, feat in enumerate(tqdm(features)):
		if int(labels[indx]) in noise_source_dict:
			source_classes = noise_source_dict[int(labels[indx])]
			source_max_score = np.max([np.abs(np.inner(singular_vector_dict[c], feat/np.linalg.norm(feat))) for c in source_classes], axis=0)
			class_score = np.abs(np.inner(singular_vector_dict[int(labels[indx])], feat/np.linalg.norm(feat)))
			scores.append(class_score - source_max_score)
		else:
			scores.append(np.abs(np.inner(singular_vector_dict[int(labels[indx])], feat/np.linalg.norm(feat))))
	return np.array(scores)

def get_score(singular_vector_dict, features, labels, normalization=True):
	'''
	Calculate the score providing the degree of showing whether the data is clean or not.
	'''
	if normalization:
		scores = [np.abs(np.inner(singular_vector_dict[int(labels[indx])], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]
	else:
		scores = [np.abs(np.inner(singular_vector_dict[int(labels[indx])], feat)) for indx, feat in enumerate(tqdm(features))]
		
	return np.array(scores)

def extract_topk(scores, labels, k):
	'''
	k: ratio to extract topk scores in class-wise manner
	To obtain the most prominsing clean data in each classes
	
	return selected labels 
	which contains top k data
	'''
	
	indexes = torch.tensor(range(len(labels)))
	selected_labels = []
	for cls in np.unique(labels):
		num = int(p * np.sum(labels==cls))
		_, sorted_idx = torch.sort(scores[labels==cls], descending=True)
		selected_labels += indexes[labels==cls][sorted_idx[:num]].numpy().tolist()
		
	return torch.tensor(selected_labels, dtype=torch.int64)

def cleansing(scores, labels):
	'''
	Assume the distribution of scores: bimodal spherical distribution.
	
	return clean labels 
	that belongs to the clean cluster made by the KMeans algorithm
	'''
	
	indexes = np.array(range(len(scores)))
	clean_labels = []
	for cls in np.unique(labels):
		cls_index = indexes[labels==cls]
		kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
		if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
			
		clean_labels += cls_index[kmeans.labels_ == 0].tolist()
		
	return np.array(clean_labels, dtype=np.int64)
		

def fine(vector_dict, current_features, current_labels, fit='kmeans', prev_features=None, prev_labels=None, p_threshold=0.5, norm=True, eigen=True):
	'''
	prev_features, prev_labels: data from the previous round
	current_features, current_labels: current round's data
	
	return clean labels
	
	if you insert the prev_features and prev_labels to None,
	the algorthm divides the data based on the current labels and current features
	
	'''
	if eigen is True:
		if prev_features is not None and prev_labels is not None:
			vector_dict = get_singular_vector(vector_dict, prev_features, prev_labels)
		else:
			vector_dict = get_singular_vector(vector_dict, current_features, current_labels)
	else:
		if prev_features is not None and prev_labels is not None:
			vector_dict = get_mean_vector(vector_dict, prev_features, prev_labels)
		else:
			vector_dict = get_mean_vector(vector_dict, current_features, current_labels)

	scores = get_score(vector_dict, features = current_features, labels = current_labels, normalization=norm)
	
	if 'kmeans' in fit:
		clean_labels = cleansing(scores, current_labels)
	elif 'gmm' in fit:
		clean_labels = fit_mixture(scores, current_labels, p_threshold=p_threshold)
	elif 'bmm' in fit:
		clean_labels = fit_mixture_bmm(scores, current_labels)
	else:
		raise NotImplemented
	
	return vector_dict, clean_labels

def fine_w_noise_source(vector_dict, noise_source_dict, clean_classes, current_features, current_labels, fit='kmeans', prev_features=None, prev_labels=None, p_threshold=0.5, norm=True, eigen=True):
	'''
	prev_features, prev_labels: data from the previous round
	current_features, current_labels: current round's data
	
	return clean labels
	
	if you insert the prev_features and prev_labels to None,
	the algorthm divides the data based on the current labels and current features
	
	'''
	if len(clean_classes) > 0:
		clean_indxs = []
		for indx, feat in enumerate(tqdm(current_features)):
			if int(current_labels[indx]) in clean_classes:
				clean_indxs.append(indx)
	else:
		clean_indxs=[]

	if eigen is True:
		if prev_features is not None and prev_labels is not None:
			vector_dict = get_singular_vector(vector_dict, prev_features, prev_labels)
		else:
			vector_dict = get_singular_vector(vector_dict, current_features, current_labels)
	else:
		if prev_features is not None and prev_labels is not None:
			vector_dict = get_mean_vector(vector_dict, prev_features, prev_labels)
		else:
			vector_dict = get_mean_vector(vector_dict, current_features, current_labels)

	scores = get_score_w_noise_source(noise_source_dict, vector_dict, features = current_features, labels = current_labels, normalization=norm)
	
	if 'kmeans' in fit:
		clean_labels = cleansing(scores, current_labels)
	elif 'gmm' in fit:
		#easy_labels, mid_labels, hard_labels = fit_mixture(scores, current_labels, clean=True, p_threshold=p_threshold)
		clean_labels = fit_mixture(scores, current_labels, clean=True, p_threshold=p_threshold)
	elif 'bmm' in fit:
		clean_labels = fit_mixture_bmm(scores, current_labels)
	else:
		raise NotImplemented

	final_clean_labels = np.array(list(set(clean_indxs)|set(clean_labels)))
	
	#return vector_dict, easy_labels, mid_labels, hard_labels
	return vector_dict, final_clean_labels

def extract_cleanidx(teacher, data_loader, parse, print_statistics = True):
	
	if parse.TFT: # 모델 불러오는 위치 통일 안됨
		teacher.load_state_dict(torch.load(parse.load_name)['state_dict'])
	else:
		teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])   
	
	teacher = teacher.cuda()

	if not parse.reinit: teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
	for params in teacher.parameters(): params.requires_grad = False
	
	if 'fine' in parse.distill_mode:
		features, labels = get_features(teacher, data_loader)
		clean_labels = fine(current_features=features, current_labels=labels, fit = parse.distill_mode)
		
	elif 'loss' in parse.distill_mode:
		clean_labels, labels = cleansing_loss(teacher, data_loader)
	else:
		raise NotImplemented
	
	if print_statistics and parse.TFT == False: return_statistics(data_loader, clean_labels)
	if parse.TFT:
		stat_dict = dict()
		df = pd.DataFrame()
		
		stat_dict['Sel_samples'] ,stat_dict['Precision'], stat_dict['Recall'], stat_dict['F1_Score'], stat_dict['Specificity'], stat_dict['Accuracy'] = return_statistics(data_loader, clean_labels)    
		df.insert(0, 'Metric', stat_dict.keys())
		df.insert(1, parse.distill_mode, stat_dict.values())
		
		root_list = parse.load_name.split('/')[:-1] + [parse.distill_mode + str(parse.dataseed) + '_statistic.csv']
		file_root = '/'.join(map(str, root_list))

		df.to_csv(file_root, index=False, header=True, sep="\t")      
		
	return clean_labels
