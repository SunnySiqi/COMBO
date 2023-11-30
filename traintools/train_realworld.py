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

from trainer import (
    Animal10NTrainer,
    Clothing1MTrainer,
    CPTrainer,
    Animal10NTrainer_2nets,
    CPTrainer_2nets
)

from utils.parse_config import ConfigParser
from utils.util import *
from utils.args import *

from custom_log import MyLogging


__all__ = ["train_realworld"]


def train_realworld(parse, config: ConfigParser):
    # By default, pytorch utilizes multi-threaded cpu
    numthread = torch.get_num_threads()
    torch.set_num_threads(numthread)
    logger = config.get_logger("train")

    # Set seed for reproducibility
    fix_seed(config["seed"])

    if config["data_loader"]["args"]["dataset"] == "CP":
        ## need convnext with adaptive interface for CP
        model = config.initialize("arch", module_arch)
    else:
        model = getattr(module_arch, "resnet50")(pretrained=True, num_classes=config["num_classes"])

    # get function handles of loss and metrics
    logger.info(config.config)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
        params=trainable_params,
        lr=config["optimizer"]["args"]["lr"],
        momentum=config["optimizer"]["args"]["momentum"],
        weight_decay=config["optimizer"]["args"]["weight_decay"],
    )

    # seperate weight decay for base parameters and proxy params
    # optimizer = torch.optim.SGD([{'params': model.params.base}, {'params': model.params.classifier, 'weight_decay':config['optimizer']['args']['proxy_weight_decay']}],
    # 	lr=config['optimizer']['args']['lr'], momentum=config['optimizer']['args']['momentum'], weight_decay= config['optimizer']['args']['weight_decay'])

    lr_scheduler = config.initialize("lr_scheduler", torch.optim.lr_scheduler, optimizer)
    if config["trainer"]["num_model"] == 1:
        if config["data_loader"]["args"]["dataset"] == "animal10N":
            trainer = Animal10NTrainer(
                model,
                metrics,
                optimizer,
                config=config,
                parse=parse,
                lr_scheduler=lr_scheduler,
            )
        elif config["data_loader"]["args"]["dataset"] == "clothing1M":
            trainer = Clothing1MTrainer(
                model,
                metrics,
                optimizer,
                config=config,
                parse=parse,
                lr_scheduler=lr_scheduler,
            )
        elif config["data_loader"]["args"]["dataset"] == "CP":
            trainer = CPTrainer(
                model,
                metrics,
                optimizer,
                config=config,
                parse=parse,
                lr_scheduler=lr_scheduler,
            )
    else:
        if config["data_loader"]["args"]["dataset"] == "CP":
            model2 = config.initialize("arch", module_arch)
        else:
            model2 = getattr(module_arch, "resnet50")(
                pretrained=True, num_classes=config["num_classes"]
            )
        trainable_params = filter(lambda p: p.requires_grad, model2.parameters())
        optimizer2 = torch.optim.SGD(
            params=trainable_params,
            lr=config["optimizer"]["args"]["lr"],
            momentum=config["optimizer"]["args"]["momentum"],
            weight_decay=config["optimizer"]["args"]["weight_decay"],
        )
        lr_scheduler2 = config.initialize("lr_scheduler", torch.optim.lr_scheduler, optimizer2)
        if config["data_loader"]["args"]["dataset"] == "animal10N":
            trainer = Animal10NTrainer_2nets(
                model,
                model2,
                metrics,
                optimizer,
                optimizer2,
                config,
                parse,
                lr_scheduler=lr_scheduler,
                lr_scheduler2=lr_scheduler2,
            )
        elif config["data_loader"]["args"]["dataset"] == "clothing1M":
            trainer = ClothingTrainer_2nets(
                model,
                model2,
                metrics,
                optimizer,
                optimizer2,
                config,
                parse,
                lr_scheduler=lr_scheduler,
                lr_scheduler2=lr_scheduler2,
            )
        elif config["data_loader"]["args"]["dataset"] == "CP":
            trainer = CPTrainer_2nets(
                model,
                model2,
                metrics,
                optimizer,
                optimizer2,
                config,
                parse,
                lr_scheduler=lr_scheduler,
                lr_scheduler2=lr_scheduler2,
            )

    trainer.train()
    trainer.logger.finish()

    logger = config.get_logger("trainer", config["trainer"]["verbosity"])
