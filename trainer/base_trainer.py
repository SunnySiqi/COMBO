from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from custom_log import MyLogging
from estimation.grow_cluster import *


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, metrics, optimizer, config, parse):
        self.config = config
        self.cls_num = self.config["num_classes"]
        dataset = config["data_loader"]["args"]["dataset"]
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.logger = MyLogging(config, project_name=parse.project_name, dataset=dataset)
        self.parse = parse
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config["n_gpu"])

        self.model = model.to(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.metrics = metrics
        self.optimizer = optimizer if type(optimizer) is not list else optimizer[0]
        #         self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer["tensorboard"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        print("-----GPU ID: ", list_ids)
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            model_name = "model_best" + str(self.config["seed"]) + ".pth"

            best_path = str(self.checkpoint_dir / model_name)
            torch.save(state, best_path)
            self.logger.info("Saving current best: " + model_name + " at: {} ...".format(best_path))

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            if metric.__name__ == "metric_cleancls":
                acc_metrics[i] += metric(output, label, self.clean_classes)
            else:
                acc_metrics[i] += metric(output, label)
            self.writer.add_scalar("{}".format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
