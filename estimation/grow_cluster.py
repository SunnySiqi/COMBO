from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture as GMM
import torch.distributions as dd
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import heapq
from collections import defaultdict

class SingleCluster(object):
    def __init__(self, startpoint):
        self.points = startpoint
        # self.centroid = np.mean(self.points, axis=0)
        self.sum_points = np.zeros_like(self.points[0]["feature"])
        self.count = 0
        for p in self.points:
            p["cluster_id"] = p["label"]
            self.sum_points += p["feature"]
            self.count += 1
        self.centroid = self.sum_points/self.count


    def add_points(self, newpoint):
        self.points.append(newpoint)
        self.sum_points += newpoint["feature"]
        self.count += 1
        self.centroid = self.sum_points/self.count


class GrowCluster(object):
    def __init__(self, cluster_startpoints, candidates):
        self.clusters = [
            SingleCluster(cluster_startpoints[i]) for i in range(len(cluster_startpoints))
        ]
        self.cluster_centroids = [self.clusters[i].centroid for i in range(len(self.clusters))]
        self.candidates = candidates

    def grow(self):
        for candidate in self.candidates:
            self.cluster_centroids = [self.clusters[i].centroid for i in range(len(self.clusters))]
            dist_to_centroids = distance.cdist(
                np.array([candidate["feature"]]), self.cluster_centroids, "euclidean"
            )
            cluster_id = np.argmin(dist_to_centroids[0])
            candidate["cluster_id"] = cluster_id
            self.clusters[cluster_id].add_points(candidate)

    def cluster_stats(self, contain_gt=True):
        all_labels = []
        all_gts = []
        all_preds = []
        all_clusterids = []
        all_features = []
        all_indexs = []
        for cluster in self.clusters:
            print("Number of Points in this cluster!!!!", len(cluster.points))

            # Use list comprehension for efficiency
            all_labels.extend([p["label"] for p in cluster.points])
            all_preds.extend([p["prediction"] for p in cluster.points])
            all_clusterids.extend([p["cluster_id"] for p in cluster.points])
            all_features.extend([p["feature"] for p in cluster.points])
            all_indexs.extend([p["index"] for p in cluster.points])
            if contain_gt:
                all_gts.extend([p["gt_label"] for p in cluster.points])

        if contain_gt:
            return all_labels, all_gts, all_preds, all_clusterids, all_features, all_indexs
        else:
            return all_labels, all_preds, all_clusterids, all_features, all_indexs

@torch.no_grad()
def get_cluster_init(model, eval_loader, device, cls_num, clean_classes):
    model.eval()
    all_indexs = []
    all_features = []
    all_labels = []
    all_gt = []
    all_predictions = []
    all_probs = []
    # with tqdm(eval_loader) as progress:
    #     for batch_idx, (data, label, indexs, gt) in enumerate(progress):
    #         data = data.to(device)
    #         all_labels.append(label)
    #         all_indexs.append(indexs)
    #         all_gt.append(gt)
    #         label = label.to(device)
    #         feat, output = model(data)
    #         all_features.append(feat.detach().cpu().numpy())
    #         outprob = torch.nn.functional.softmax(output).data.detach()
    #         y_prob, y_pred = outprob.max(1)
    #         all_predictions.append(y_pred.detach().cpu().numpy())
    #         all_probs.append(y_prob.detach().cpu().numpy())
    # all_labels = np.hstack(all_labels)
    # all_indexs = np.hstack(all_indexs)
    # all_gt = np.hstack(all_gt)
    # all_predictions = np.hstack(all_predictions)
    # all_probs = np.hstack(all_probs)
    # all_features = np.concatenate(all_features, axis=0)

    with torch.no_grad(), tqdm(eval_loader) as progress:
        for batch_idx, (data, label, indexs, gt) in enumerate(progress):
            data = data.to(device)
            all_labels.append(label)
            all_indexs.append(indexs)
            all_gt.append(gt)
            feat, output = model(data)
            all_features.append(feat.detach().cpu())
            outprob = torch.nn.functional.softmax(output).data.detach()
            y_prob, y_pred = outprob.max(1)
            all_predictions.append(y_pred.detach().cpu())
            all_probs.append(y_prob.detach().cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_indexs = torch.cat(all_indexs).numpy()
    all_gt = torch.cat(all_gt).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_features = torch.cat(all_features, axis=0).numpy()

    # all_samples = {}
    # label_correct_sample_dict = {}
    # label_sample_dict = {}
    # for i in range(len(all_indexs)):
    #     sample = {}
    #     sample["index"] = all_indexs[i]
    #     sample["feature"] = all_features[i]
    #     sample["gt_label"] = all_gt[i]
    #     sample["label"] = all_labels[i]
    #     sample["prediction"] = all_predictions[i]
    #     sample["prob"] = all_probs[i]
    #     all_samples[sample["index"]] = sample

    #     if sample["label"] in label_sample_dict:
    #         label_sample_dict[sample["label"]].append(sample)
    #     else:
    #         label_sample_dict[sample["label"]] = [sample]

    #     if sample["prediction"] == sample["label"]:
    #         if sample["label"] in label_correct_sample_dict:
    #             label_correct_sample_dict[sample["label"]].append(sample)
    #         else:
    #             label_correct_sample_dict[sample["label"]] = [sample]


    all_samples = {index: {"index": index, "feature": feature, "gt_label": gt_label, "label": label, "prediction": prediction, "prob": prob} 
                   for index, feature, gt_label, label, prediction, prob in zip(all_indexs, all_features, all_gt, all_labels, all_predictions, all_probs)}

    label_sample_dict = defaultdict(list)
    label_correct_sample_dict = defaultdict(list)

    for sample in all_samples.values():
        label_sample_dict[sample["label"]].append(sample)
        if sample["prediction"] == sample["label"]:
            label_correct_sample_dict[sample["label"]].append(sample)


    start_points = []
    start_points_idxs = []
    candidates = []
    for i in range(cls_num):
        if i in clean_classes:
            start = label_sample_dict[i]
        elif i in label_correct_sample_dict:
            # label_correct_sample_dict[i] = sorted(
            #     label_correct_sample_dict[i], key=lambda x: x["prob"], reverse=True
            # )
            cut = int(0.1 * len(label_correct_sample_dict[i])) + 1
            label_correct_sample_dict[i] = heapq.nlargest(cut, label_correct_sample_dict[i], key=lambda x: x["prob"])
            start = label_correct_sample_dict[i][:cut]
        else:
            # label_sample_dict[i] = sorted(label_sample_dict[i], key=lambda x: x["prob"])
            # cut = int(0.05 * len(label_sample_dict[i])) + 1  ## TODO: try heapq to speed up
            ## use heapq to speed up
            cut = int(0.05 * len(label_sample_dict[i])) + 1 
            label_sample_dict[i] = heapq.nsmallest(cut, label_sample_dict[i], key=lambda x: x["prob"])
            

            start = label_sample_dict[i][:cut]
        start_points.append(start)
        for p in start:
            start_points_idxs.append(p["index"])

    start_points_idxs = set(start_points_idxs)
    ## remove all these start_points_idxs keys from all_samples
    all_samples = {k: v for k, v in all_samples.items() if k not in start_points_idxs}

    # for i in all_samples:
    #     if i not in start_points_idxs:
    #         candidates.append(all_samples[i])
    candidates = list(all_samples.values())
    candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    return start_points, candidates
