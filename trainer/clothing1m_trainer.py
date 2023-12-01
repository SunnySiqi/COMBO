import os
from estimation import *
from detection.learn_noise_sources import *
from detection.selection import *
from detection.UNICON import *
import dataloader.Clothing1M as dataloader
from loss.semi_loss import *
from loss.tv_loss import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from trainer import BaseTrainer
from utils.util import inf_loop
import sys
import numpy as np
import pickle
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
# torch.multiprocessing.set_sharing_strategy("file_descriptor")


def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm


class Clothing1MTrainer(BaseTrainer):
    """
    DefaultTrainer class

    Note:
            Inherited from BaseTrainer.
    """

    def __init__(self, model, metrics, optimizer, config, parse, lr_scheduler=None, len_epoch=None):
        super().__init__(model, metrics, optimizer, config, parse)
        self.config = config
        self.parse = parse
        self.num_imgs = config["data_loader"]["args"]["num_imgs"]

        batch_size = self.config["data_loader"]["args"][
            "batch_size"
        ]  ## TODO: change this to config
        num_workers = self.config["data_loader"]["args"]["num_workers"]
        root = config["data_loader"]["args"]["data_dir"]
        warmup_batch_size = 4 * batch_size
        self.loader = dataloader.clothing1M_dataloader(
            root, batch_size, warmup_batch_size, num_workers, num_imgs=self.num_imgs
        )

        self.warm_up_loader = self.loader.run(
            "warmup", supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]
        )
        self.eval_loader = self.loader.run(
            "eval_train", supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]
        )
        self.val_loader = self.loader.run(
            "val", supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]
        )
        self.test_loader = self.loader.run(
            "test", supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]
        )

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(batch_size))

        self.val_criterion = CrossEntropyLoss()
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []

        self.clean_idx = None
        self.noise_source_dict = {}
        self.noise_sources = set()
        self.clean_classes = np.array([])

        self.estimation_method = self.config["estimation"]["method"]
        # self.estimate_every = self.config['estimation']['every']
        self.detection_method = self.config["detection"]["method"]
        self.update_dataloader_every = self.config["detection"]["every"]
        self.train_method = self.config["trainer"]["train_noise_method"]

        if self.estimation_method == "total_variation":
            self.transition = dirichlet_transition(
                self.device, self.cls_num, 100, 1e-32, (0.999, 0.01)
            )
            self.regularization = tv_regularization(
                num_pairs=self.config["data_loader"]["args"]["batch_size"]
            )
            self.gamma = 0.1
            self.train_criterion_tv = TVLoss()
        elif self.estimation_method == "robot":
            self.loss_name = "forward"
            self.outer_obj = "rce"
            self.T_init = 4.5
            self.trans = sig_t(self.device, self.cls_num, init=self.T_init)
            self.trans = self.trans.to(self.device)
            self.meta_optimizer = torch.optim.SGD(self.trans.parameters(), lr=1e-1, weight_decay=0, momentum=0.9)
        
        if self.detection_method == "FINE+K":
            self.vector_dict = {}
        elif self.detection_method == "UNICON+K":
            self.d_u = self.config["detection"]["UNICON+K"]["d_u"]
            self.tau = self.config["detection"]["UNICON+K"]["tau"]
        
        if self.train_method == "SSL":
            self.train_criterion = SemiLoss()
            self.train_contrastive_criterion = SupConLoss()
        elif self.train_method == "unicon":
            self.train_criterion_unicon = SemiLoss()
            self.train_contrastive_criterion_unicon = SupConLoss()
            self.T = self.config["trainer"]["UNICON+K"]["T"]

    def update_dataloader(
        self, epoch, current_features, current_labels, idxs, cluster_labels=None, selected_idx=None
    ):
        current_features = np.array(current_features)
        current_labels = np.array(current_labels)
        if cluster_labels is not None:
            cluster_labels = np.array(cluster_labels)
        idxs = np.array(idxs)

        if self.detection_method == "FINE+K":
            if self.clean_idx is not None:
                prev_features, prev_labels = (
                    current_features[self.clean_idx],
                    current_labels[self.clean_idx],
                )
            else:
                prev_features, prev_labels = current_features, current_labels
            self.vector_dict, self.clean_idx = fine_w_noise_source(
                self.vector_dict,
                self.noise_source_dict,
                self.clean_classes,
                current_features,
                current_labels,
                fit="fine-gmm",
                prev_features=prev_features,
                prev_labels=prev_labels,
                p_threshold=0.5,
            )
        elif self.detection_method == "UNICON+K":
            self.clean_idx = selected_idx
        self.clean_idx = idxs[self.clean_idx]
        self.noisy_idx = np.delete(idxs, self.clean_idx)

        self.labeled_dataloader, self.unlabeled_dataloader = self.loader.run(
            "train",
            supervised_idxs=self.clean_idx,
            semi_supervised_idxs=self.noisy_idx,
            cluster_labels=cluster_labels,
        )

        # self.selected, self.precision, self.recall, self.f1, self.specificity, self.accuracy = return_statistics(self.eval_loader, self.clean_idx)

    def _train_epoch_classifer_consistent(self, model, optimizer, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        with tqdm(self.warm_up_loader) as progress:
            for batch_idx, (data, label, indexs, _) in enumerate(progress):
                progress.set_description_str(f"Train epoch {epoch}")
                data, label = data.to(self.device), label.long().to(self.device)

                model_represent, output = model(data)

                if self.estimation_method == "total_variation":
                    loss = self.train_criterion_tv(
                        output, label, self.gamma, self.transition, self.regularization
                    )
                    self.transition.update(output, label)
                elif self.estimation_method == "dualT":
                    probs = F.softmax(output, dim=1)
                    output = torch.matmul(probs, torch.tensor(self.t_m).float().to(self.device))
                    loss = torch.nn.functional.cross_entropy(output, label)
                elif self.estimation_method == "robot":
                    prob = F.softmax(output, dim=1)
                    prob = prob.t()
                    loss, out_softmax, p_T = foward_loss(output, label, self.trans().detach())
                    out_forward = torch.matmul(self.trans().t(), prob)
                    out_forward = out_forward.t()
                    output = out_forward

                else:
                    loss = torch.nn.functional.cross_entropy(output, label)
                optimizer.zero_grad()

                # loss.backward(retain_graph=True)
                loss.backward()

                optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar("loss", loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(self._progress(batch_idx), loss.item())
                    )
                    self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))
        return total_loss

    def _train_epoch_sample_selection(self, model, optimizer, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        with tqdm(self.labeled_dataloader) as progress:
            for batch_idx, (data, _, _, _, label) in enumerate(progress):
                progress.set_description_str(f"Train epoch {epoch}")
                data, label = data.to(self.device), label.long().to(self.device)

                model_represent, output = model(data)

                if self.estimation_method == "total_variation":
                    loss = self.train_criterion_tv(
                        output, label, self.gamma, self.transition, self.regularization
                    )
                    self.transition.update(output, label)
                else:
                    loss = torch.nn.functional.cross_entropy(output, label)
                optimizer.zero_grad()

                loss.backward(retain_graph=True)

                optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar("loss", loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(self._progress(batch_idx), loss.item())
                    )
                    self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))
        return total_loss

    def _train_epoch_unicon(self, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        num_iter = (
            len(self.labeled_dataloader.dataset) // self.config["data_loader"]["args"]["batch_size"]
        ) + 1

        JS_dist = Jensen_Shannon()
        unlabeled_train_iter = iter(self.unlabeled_dataloader)

        with tqdm(self.labeled_dataloader) as progress:
            for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x) in enumerate(
                progress
            ):
                try:
                    (
                        inputs_u,
                        inputs_u2,
                        inputs_u3,
                        inputs_u4,
                        labels_u,
                    ) = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(self.unlabeled_dataloader)
                    (
                        inputs_u,
                        inputs_u2,
                        inputs_u3,
                        inputs_u4,
                        labels_u,
                    ) = unlabeled_train_iter.next()

                batch_size = inputs_x.size(0)
                label = labels_x.cuda()
                labels_x = torch.zeros(batch_size, self.cls_num).scatter_(
                    1, labels_x.view(-1, 1), 1
                )

                inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x = (
                    inputs_x.cuda(),
                    inputs_x2.cuda(),
                    inputs_x3.cuda(),
                    inputs_x4.cuda(),
                    labels_x.cuda(),
                )
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = (
                    inputs_u.cuda(),
                    inputs_u2.cuda(),
                    inputs_u3.cuda(),
                    inputs_u4.cuda(),
                    labels_u.cuda(),
                )

                with torch.no_grad():
                    # Label guessing of unlabeled samples
                    _, outputs_u11 = self.model(inputs_u)
                    _, outputs_u12 = self.model(inputs_u2)

                    ## Pseudo-label
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2

                    ptu = pu ** (1 / self.T)  ## Temparature Sharpening

                    targets_u = (ptu / ptu.sum(dim=1, keepdim=True)).detach()

                    ## Label refinement
                    _, outputs_x = self.model(inputs_x)
                    _, outputs_x2 = self.model(inputs_x2)

                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    # one_hot = F.one_hot(labels_x.to(torch.int64), num_classes = self.cls_num)
                    w_x = 1 - JS_dist(px, labels_x)

                    px = w_x.unsqueeze(1) * labels_x + (1 - w_x).unsqueeze(1) * px
                    ptx = px ** (1 / self.T)  ## Temparature sharpening

                    targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

                ## Unsupervised Contrastive Loss
                f1, _ = self.model(inputs_u3)
                f2, _ = self.model(inputs_u4)
                f1 = torch.nn.functional.normalize(f1, dim=1).unsqueeze_(1)
                f2 = torch.nn.functional.normalize(f2, dim=1).unsqueeze_(1)
                features = torch.cat([f1, f2], dim=1)

                loss_simCLR = self.train_contrastive_criterion_unicon(features)

                # MixMatch
                l = np.random.beta(4, 4)
                l = max(l, 1 - l)
                all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                ## Mixup
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                _, logits = self.model(mixed_input)
                logits_x = logits[: batch_size * 2]
                logits_u = logits[batch_size * 2 :]

                ## Combined Loss
                Lx, Lu, lamb = self.train_criterion_unicon(
                    logits_x,
                    mixed_target[: batch_size * 2],
                    logits_u,
                    mixed_target[batch_size * 2 :],
                    epoch + batch_idx / num_iter,
                    self.config["trainer"]["warmup"],
                )

                ## Regularization
                prior = torch.ones(self.cls_num) / self.cls_num
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                ## Total Loss
                loss = Lx + lamb * Lu + 0.5 * loss_simCLR + penalty

                if self.estimation_method == "total_variation":
                    loss += self.train_criterion_tv(
                        px, label, self.gamma, self.transition, self.regularization
                    )
                    self.transition.update(px, label)

                ## Accumulate Loss
                # loss_x += Lx.item()
                # loss_u += Lu.item()
                # loss_ucl += loss_simCLR.item()

                # Compute gradient and Do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar("loss", loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                px = px.unsqueeze(1)
                total_metrics += self._eval_metrics(px, labels_x)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(self._progress(batch_idx), loss.item())
                    )
        return total_loss

    def _train_epoch_cluster_label(self, model, optimizer, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        num_iter = (
            len(self.labeled_dataloader.dataset) // self.config["data_loader"]["args"]["batch_size"]
        ) + 1
        unlabeled_train_iter = iter(self.unlabeled_dataloader)
        with tqdm(self.labeled_dataloader) as progress:
            for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x) in enumerate(
                progress
            ):
                try:
                    (
                        _,
                        _,
                        inputs_u3,
                        inputs_u4,
                        labels_u,
                    ) = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(self.unlabeled_dataloader)
                    (
                        _,
                        _,
                        inputs_u3,
                        inputs_u4,
                        labels_u,
                    ) = unlabeled_train_iter.next()

                batch_size = inputs_x.size(0)
                labels_x = labels_x.to(self.device)
                label = labels_x
                labels_x = torch.zeros(batch_size, self.cls_num, device=self.device).scatter_(
                    1, labels_x.view(-1, 1), 1
                )
                labels_u = torch.zeros(batch_size, self.cls_num, device=self.device).scatter_(
                    1, labels_u.view(-1, 1).to(self.device), 1
                )

                inputs_x, inputs_x2, inputs_x3, inputs_x4 = (
                    inputs_x.to(self.device),
                    inputs_x2.to(self.device),
                    inputs_x3.to(self.device),
                    inputs_x4.to(self.device),
                )
                inputs_u3, inputs_u4 = (inputs_u3.to(self.device), inputs_u4.to(self.device))

                targets_x = (labels_x / labels_x.sum(dim=1, keepdim=True)).detach()
                targets_u = (labels_u / labels_u.sum(dim=1, keepdim=True)).detach()

                ## Unsupervised Contrastive Loss
                f1, _ = model(inputs_u3)
                f2, _ = model(inputs_u4)
                f1 = torch.nn.functional.normalize(f1, dim=1).unsqueeze_(1)
                f2 = torch.nn.functional.normalize(f2, dim=1).unsqueeze_(1)
                features = torch.cat([f1, f2], dim=1)

                loss_simCLR = self.train_contrastive_criterion(features)

                # MixMatch
                l = np.random.beta(4, 4)
                l = max(l, 1 - l)
                all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                ## Mixup
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                _, logits = model(mixed_input)
                logits_x = logits[: batch_size * 2]
                logits_u = logits[batch_size * 2 :]

                ## Combined Loss
                Lx, Lu, lamb = self.train_criterion(
                    logits_x,
                    mixed_target[: batch_size * 2],
                    logits_u,
                    mixed_target[batch_size * 2 :],
                    epoch + batch_idx / num_iter,
                    self.config["trainer"]["warmup"],
                )

                ## Regularization
                prior = torch.ones(self.cls_num, device=self.device) / self.cls_num
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                # superived label loss
                f_x1, outputs_x1 = model(inputs_x)
                f_x2, outputs_x3 = model(inputs_x3)
                f_x1 = torch.nn.functional.normalize(f_x1, dim=1).unsqueeze_(1)
                f_x2 = torch.nn.functional.normalize(f_x2, dim=1).unsqueeze_(1)
                features = torch.cat([f_x1, f_x2], dim=1)

                supervised_loss_simCLR = self.train_contrastive_criterion(features)
                supervised_loss = torch.nn.functional.cross_entropy(
                    outputs_x1, labels_x
                ) + torch.nn.functional.cross_entropy(outputs_x3, labels_x)
                ## Total Loss
                loss = (
                    Lx
                    + lamb * Lu
                    + 0.5 * loss_simCLR
                    + 0.5 * supervised_loss_simCLR
                    + supervised_loss
                    + penalty
                )

                ## Accumulate Loss
                # loss_x += Lx.item()
                # loss_u += Lu.item()
                # loss_ucl += loss_simCLR.item()

                # Compute gradient and Do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # total_loss += loss
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar("loss", loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()

                total_metrics += self._eval_metrics(outputs_x1, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(self._progress(batch_idx), loss.item())
                    )
        return total_loss

    def _train_epoch(self, epoch):
        """

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
                If you have additional information to record, for example:
                        > additional_log = {"x": x, "y": y}
                merge it with log before return. i.e.
                        > log = {**log, **additional_log}
                        > return log

                The metrics in log must have the key 'metrics'.
        """
        if self.detection_method == "none":
            num_iter = (
                len(self.warm_up_loader.dataset) // self.config["data_loader"]["args"]["batch_size"]
            ) + 1
        else:
            num_iter = (
                len(self.labeled_dataloader.dataset)
                // self.config["data_loader"]["args"]["batch_size"]
            ) + 1
        self.model.train()

        # train epoch
        if self.detection_method == "none":
            total_loss = self._train_epoch_classifer_consistent(self.model, self.optimizer, epoch)

        elif self.train_method == "SSL":
            total_loss = self._train_epoch_cluster_label(self.model, self.optimizer, epoch)

        elif self.train_method == "unicon":
            total_loss = self._train_epoch_unicon(epoch)

        elif self.train_method == "none":
            total_loss = self._train_epoch_sample_selection(self.model, self.optimizer, epoch)

        # self.purity = (self.labeled_dataloader.dataset.noise_label == \
        #                  self.labeled_dataloader.dataset.train_labels_gt).sum() / \
        #               len(self.labeled_dataloader.dataset.noise_label)
        log = {
            "loss": total_loss / num_iter,
            "learning rate": self.lr_scheduler.get_last_lr(),
            # 'purity:': '{} = {}/{}'.format(self.purity, (self.labeled_dataloader.dataset.noise_label == \
            #      self.labeled_dataloader.dataset.train_labels_gt).sum(), len(self.labeled_dataloader.dataset.noise_label))
        }

        val_log = self._valid_epoch(epoch)
        log.update(val_log)

        test_log = self._test_epoch(epoch)
        log.update(test_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
                The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        all_labels = []
        all_preds = []
        with torch.no_grad():
            with tqdm(self.val_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f"Valid epoch {epoch}")
                    data, label = data.to(self.device), label.long().to(self.device)
                    _, output = self.model(data)
                    loss = self.val_criterion(output, label)
                    all_labels += label.cpu().detach().numpy().tolist()
                    _, y_pred = output.max(1)
                    all_preds += y_pred.cpu().detach().numpy().tolist()

                    self.writer.set_step((epoch - 1) * len(self.val_loader) + batch_idx, "valid")
                    self.writer.add_scalar("loss", loss.item())
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        # 	self.writer.add_histogram(name, p, bins='auto')

        cm = confusion_matrix(all_labels, all_preds)
        print("!!!!!VALIDATION CONFUSION MATRIX!!!!!")
        print(cm)
        print("-------------------------------------")

        plt.figure(figsize=(12, 7))
        cm_fig = sn.heatmap(cm).get_figure()
        self.writer.add_figure("Validation Confusion matrix", cm_fig, epoch)

        return {
            "val_loss": total_val_loss / len(self.val_loader),
            "val_metrics": (total_val_metrics / len(self.val_loader)).tolist(),
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
                The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        all_labels = []
        all_preds = []
        # results = np.zeros((len(self.test_loader.dataset.test_data), self.config['num_classes']), dtype=np.float32)
        # tar_ = np.zeros((len(self.test_loader.dataset.test_data),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f"Test epoch {epoch}")
                    data, label = data.to(self.device), label.long().to(self.device)
                    _, output = self.model(data)
                    loss = self.val_criterion(output, label)
                    all_labels += label.cpu().detach().numpy().tolist()
                    _, y_pred = output.max(1)
                    all_preds += y_pred.cpu().detach().numpy().tolist()

                    self.writer.set_step((epoch - 1) * len(self.test_loader) + batch_idx, "test")
                    self.writer.add_scalar("loss", loss.item())
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

                    # results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                    # tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        # 	self.writer.add_histogram(name, p, bins='auto')

        cm = confusion_matrix(all_labels, all_preds)
        print("!!!!!TEST CONFUSION MATRIX!!!!!")
        print(cm)
        print("-------------------------------------")

        plt.figure(figsize=(12, 7))
        cm_fig = sn.heatmap(cm).get_figure()
        self.writer.add_figure("Test Confusion matrix", cm_fig, epoch)

        return {
            "test_loss": total_test_loss / len(self.test_loader),
            "test_metrics": (total_test_metrics / len(self.test_loader)).tolist(),
        }
        # },[results,tar_]

    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()
        self.len_epoch = len(self.warm_up_loader)
        with tqdm(self.warm_up_loader) as progress:
            for batch_idx, (data, label, indexs, _) in enumerate(progress):
                progress.set_description_str(f"Warm up epoch {epoch}")

                data, label = data.to(self.device), label.long().to(self.device)

                self.optimizer.zero_grad()
                _, output = self.model(data)
                # out_prob = torch.nn.functional.softmax(output).data.detach()

                # self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

                loss = torch.nn.functional.cross_entropy(output, label)

                loss.backward(retain_graph=True)
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar("loss", loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(
                        " {} Loss: {:.6f}".format(self._progress(batch_idx), loss.item())
                    )
                    self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        # self.warmup_memorybank = self.get_prob_memorybank()
        log = {
            "loss": total_loss / self.len_epoch,
            "noise detection rate": 0.0,
            "metrics": (total_metrics / self.len_epoch).tolist(),
            "learning rate": self.lr_scheduler.get_last_lr(),
        }

        val_log = self._valid_epoch(epoch)
        log.update(val_log)

        test_log = self._test_epoch(epoch)
        log.update(test_log)

        return log

    def train(self):
        not_improved_count = 0
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc="Total progress: "):
            # check the clusters in feature space of std training
            if epoch <= self.config["trainer"]["warmup"]:
                if (
                    self.config["data_loader"]["args"]["dataset"] == "Clothing1M"
                    and self.num_imgs <= 300000
                ):
                    ## set random seed for random
                    random.seed(epoch + 2024)
                    self.warm_up_loader = self.loader.run(
                        "warmup", supervised_idxs=[], semi_supervised_idxs=[], cluster_labels=[]
                    )
                result = self._warmup_epoch(epoch)
            elif (
                epoch == self.config["trainer"]["warmup"] + 1
                or epoch % self.update_dataloader_every == 1
                # Estimation to get matrix
            ):
                if self.estimation_method != "none":
                    # Estimation to get matrix
                    (
                        source_matrix,
                        all_labels,
                        all_features,
                        all_indexs,
                        all_clusterids,
                    ) = self.get_estimation_matrix(self.model, epoch=epoch)
                    # Learn noise source
                    self.get_noise_sources(source_matrix)
                else:
                    all_labels, all_features, all_indexs = self.get_feature(self.model)
                    all_clusterids = None
                # Update dataloader
                if self.detection_method != "none":
                    selected_idx = None
                    if self.detection_method == "UNICON+K":
                        selected_idx = select_idx_UNICON(
                            self.model,
                            self.eval_loader,
                            self.noise_source_dict,
                            self.clean_classes,
                            self.d_u,
                            self.tau,
                            self.cls_num,
                        )
                    self.update_dataloader(
                        epoch, all_features, all_labels, all_indexs, all_clusterids, selected_idx
                    )
                result = self._train_epoch(epoch)
            else:
                # if self.config["data_loader"]["args"]["dataset"] == "Clothing1M" and self.num_imgs <= 300000:
                #     ## set random seed for random
                #     random.seed(epoch+2024)
                #     cluster_labels = np.array(all_clusterids) if all_clusterids is not None else None
                #     self.labeled_dataloader, self.unlabeled_dataloader = self.loader.run(
                #         "train",
                #         supervised_idxs=self.clean_idx,
                #         semi_supervised_idxs=self.noisy_idx,
                #         cluster_labels=cluster_labels,
                #     )
                result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            for key, value in result.items():
                if key == "metrics":
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == "metrics_gt":
                    log.update(
                        {mtr.__name__ + "_gt": value[i] for i, mtr in enumerate(self.metrics)}
                    )
                elif key == "val_metrics":
                    log.update(
                        {"val_" + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}
                    )
                elif key == "test_metrics":
                    log.update(
                        {"test_" + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}
                    )
                else:
                    log[key] = value

            self.logger.log(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(self.mnt_metric)
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.logger.log({"BEST_test_metric_overall": log["test_metric_overall"]})

                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def get_estimation_matrix(self, model, epoch: Optional[int] = None):
        if self.estimation_method == "growing_cluster":
            start_points, candidates = get_cluster_init(
                model, self.eval_loader, self.device, self.cls_num, self.clean_classes
            )
            growing_cluster = GrowCluster(start_points, candidates)
            growing_cluster.grow()
            (
                all_labels,
                all_preds,
                all_clusterids,
                all_features,
                all_indexs,
            ) = growing_cluster.cluster_stats(contain_gt=False)
            print("!!!!!TRAIN KMEANS CLUSTER GT CONFUSION MATRIX!!!!!")
            print("!!!!!TRAIN KMEANS CLUSTER NOISY LABEL CONFUSION MATRIX!!!!!")
            label_cluster_cm = confusion_matrix(all_labels, all_clusterids)
            print(label_cluster_cm)
            print("-------------------------------------")
            source_matrix = label_cluster_cm
        elif self.estimation_method == "dualT":
            T_spadesuit, T_clubsuit, all_labels, all_indexs, all_features = get_transition_matrices(
                model, self.cls_num, self.eval_loader, self.device
            )
            source_matrix = np.matmul(T_clubsuit, T_spadesuit)
            self.t_m = source_matrix
            # print("DUALT T MATRIX!!!!!!!", self.t_m)
            all_clusterids = None
        elif self.estimation_method == "total_variation":
            source_matrix = self.transition.concentrations.cpu().detach().numpy()
            print("VT SOURCE MATRIX", source_matrix)
            all_labels, all_features, all_indexs = self.get_feature(model)
            all_clusterids = None
        elif self.estimation_method == "robot":
            source_matrix = robot_update_t(
                self.trans,
                self.eval_loader,
                self.device,
                model,
                self.meta_optimizer,
                self.cls_num,
                self.loss_name,
                self.outer_obj,
            )
            all_labels, all_features, all_indexs = self.get_feature(model)
            all_clusterids = None

        return source_matrix, all_labels, all_features, all_indexs, all_clusterids

    def get_noise_sources(self, source_matrix):
        self.noise_source_dict = {}
        self.noise_sources = set()
        for i in range(self.cls_num):
            cluster_distribution = np.array(source_matrix[:, i]).reshape(-1, 1)
            # Use GMM to select noise sources: 2/3 distributions
            noise_sources = noise_sources_from_GMM(cluster_distribution)
            if i in list(noise_sources):
                noise_sources = list(noise_sources)
                noise_sources.remove(i)
                noise_sources = np.array(noise_sources)
            if len(noise_sources) == 0:
                continue
            for noise in noise_sources:
                if noise in self.noise_source_dict:
                    self.noise_source_dict[noise].append(i)
                else:
                    self.noise_source_dict[noise] = [i]
            self.noise_sources.update([i])
            self.clean_classes = set(np.arange(self.cls_num)).difference(
                set(list(self.noise_source_dict.keys()))
            )
        print("NOISE SOURCE DICT!!!!!!!")
        print(self.noise_source_dict)
        print("NOISE SOURCES!!!!!!!")
        print(self.noise_sources)
        print("CLEAN CLASSES!!!!!!!!")
        print(self.clean_classes)

    @torch.inference_mode()
    def get_feature(self, model):
        model.eval()
        all_labels = []
        all_indexs = []
        all_features = []
        with tqdm(self.eval_loader) as progress:
            for _, (data, label, indexs, _) in enumerate(progress):
                data = data.to(self.device)
                all_labels.append(label)
                all_indexs.append(indexs)
                feat, _ = model(data)
                all_features.append(feat)
        all_labels = np.hstack(all_labels)
        all_indexs = np.hstack(all_indexs)
        all_features = torch.cat(all_features, dim=0)
        all_features = all_features.cpu().numpy()
        return all_labels, all_features, all_indexs

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
