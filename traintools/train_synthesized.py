import sys
import requests
import socket
import random
import numpy as np
import copy
import torch

from collections import OrderedDict
import loss as module_loss

import model.metric as module_metric
import model.model as module_arch

from trainer import BaseTrainer, SynthesizedTrainer, SynthesizedTrainer_2nets

from utils.parse_config import ConfigParser
from utils.util import *
from utils.args import *

from custom_log import MyLogging


__all__ = ["train_synthesized"]


def train_synthesized(parse, config: ConfigParser):
    # By default, pytorch utilizes multi-threaded cpu
    numthread = torch.get_num_threads()
    torch.set_num_threads(numthread)
    logger = config.get_logger("train")

    # Set seed for reproducibility
    fix_seed(config["seed"])

    # build model architecture, then print to console
    model = config.initialize("arch", module_arch)

    # get function handles of loss and metrics
    logger.info(config.config)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])
    # seperate weight decay for base parameters and proxy params
    optimizer = torch.optim.SGD(
        [
            {"params": model.params.base},
            {
                "params": model.params.classifier,
                "weight_decay": config["optimizer"]["args"]["proxy_weight_decay"],
            },
        ],
        lr=config["optimizer"]["args"]["lr"],
        momentum=config["optimizer"]["args"]["momentum"],
        weight_decay=config["optimizer"]["args"]["weight_decay"],
    )

    lr_scheduler = config.initialize("lr_scheduler", torch.optim.lr_scheduler, optimizer)
    trainer = SynthesizedTrainer(
            model,
            metrics,
            optimizer,
            config=config,
            parse=parse,
            lr_scheduler=lr_scheduler,
    )

    trainer.train()
    trainer.logger.finish()

    logger = config.get_logger("trainer", config["trainer"]["verbosity"])
