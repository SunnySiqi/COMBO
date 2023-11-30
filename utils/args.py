import argparse
import collections
import os
from .parse_config import ConfigParser
from collections import OrderedDict

def parse_args():

    args = argparse.ArgumentParser(description='PyTorch Template')
    
    args.add_argument('-c', '--config', 
                      default=None, 
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', 
                      '--resume', 
                      default=None, 
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', 
                      '--device', 
                      default='0', 
                      type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('--no_wandb', 
                      action='store_false', 
                      help='if false, not to use wandb')
    args.add_argument('--traintools', 
                      type=str, 
                      default='train_synthesized', 
                      choices=['train_synthesized', 'train_realworld'])
    args.add_argument("--sccid",  type=str,  default=None, help="scc job id")
    args.add_argument("--project_name",  type=str,  default="learning_with_noise_v1", help="project name for wandb")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--dataset', '--ds'], type=str, target=('data_loader', 'args', 'dataset')),
        CustomArgs(['--data_dir', '--dir'], type=str, target=('data_loader', 'args', 'data_dir')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--warmup'], type=int, target=('trainer','warmup')),
        CustomArgs(['--noise_ratio', '--percent'], type=float, target=('data_loader', 'noise', 'percent')),
        CustomArgs(['--noise_type', '--noise'], type=str, target=('data_loader', 'noise', 'type')),
        CustomArgs(['--estimation_method'], type=str, target=('estimation', 'method')),
        CustomArgs(['--detection_method'], type=str, target=('detection', 'method')),
        CustomArgs(['--train_noise_method'], type=str, target=('trainer', 'train_noise_method')),
        CustomArgs(['--num_model'], type=int, target=('trainer', 'num_model')),
        CustomArgs(['--num_imgs'], type=int, target=('data_loader', 'args', 'num_imgs')),
        CustomArgs(['--every'], type=int, target=('detection', 'every')),
    ]
    
    config = ConfigParser.get_instance(args, options)
    parse = args.parse_args()
    
    return config, parse



def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)