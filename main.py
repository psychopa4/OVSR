import sys
import os
from os.path import join, exists
from functools import reduce
import time
import yaml
import importlib
import random
import argparse
import torch
import torch.multiprocessing as mp
from ptflops import get_model_complexity_info
from util import automkdir, DICT2OBJ
from train import train, test


def ddp_func(demo_fn, config):
    mp.spawn(demo_fn,
             args=(config,),
             nprocs=len(config.gpus),
             join=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, default='./options/ovsr.yml')
    cfg = parser.parse_args()

    with open(cfg.options, 'r', encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file.read())
        config = DICT2OBJ(config)
        
    gpus = config.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = reduce(lambda x, y: str(x) + ', ' + str(y), gpus[1:], str(gpus[0]) if len(gpus) > 0 else '-1')
    config.seed = random.randint(1, 10000)

    config.path.checkpoint = join(config.path.base, config.path.checkpoint, config.model.name)
    config.path.eval_result = join(config.path.base, config.path.eval_result, config.model.name)
    config.path.resume = join(config.path.checkpoint, f'{config.train.resume:04}.pth')
    config.path.eval_file = join(config.path.base, f'eval_{config.model.name}.txt')
    automkdir(config.path.checkpoint)
    automkdir(config.path.eval_result)

    config.network = importlib.import_module(f'models.{config.model.file}').Net(config)

    if config.function == 'get_complexity':
        macs, params = get_model_complexity_info(config.network, (3, 1, 180, 320), as_strings=False,
                                                print_per_layer_stat=False, verbose=True, ignore_modules=[torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.PReLU])
        print('Computational complexity: {:,}'.format(macs))
        print('Number of parameters: {:,}'.format(params))
        exit()

    function = getattr(sys.modules[__name__], config.function)

    try:
        ddp_func(function, config)
    except RuntimeWarning as e:
        print(e)