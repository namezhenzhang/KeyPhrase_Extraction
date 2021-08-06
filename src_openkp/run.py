# from operator import add
from re import L, VERBOSE
# from numpy.lib.function_base import flip

from torch import optim
import logger
import os
from configs import get_args_parser, get_args, save_args

import torch.nn as nn
from tqdm import tqdm
from datasets import OpenKPDataset
# from optimizer import get_optimizer
import random
import torch
import numpy  as np
from model import get_model
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
# from sklearn.metrics import f1_score, accuracy_score
# from criterion import MultiClassCriterion, evaluate, get_preds_from_logits, get_label_words_logits
from transformers import (RobertaConfig, RobertaForMaskedLM,RobertaModel, RobertaTokenizer)

args = get_args_parser()
save_args()
logger.setup_applevel_logger(logger_name='kp', file_name=args.logger_file_name)
log = logger.get_logger(__name__)
# log a few args
log.info("unicode: {}".format(args.random_code))
log.info("ngpu: {}".format(args.n_gpu))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_tokenizer(special=[]):
    args = get_args()
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base",
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer

class Runner(object):
    def __init__(self, args,special=[]):
        super(Runner,self).__init__()
        self.args = args
        self.tokenizer = get_tokenizer(special=special)
        self.datasetclass = OpenKPDataset

    def __get_testdataset(self):
        test_dataset = self.datasetclass(args=self.args, tokenizer=self.tokenizer, split="test")
        test_dataset.cuda()
        self.test_dataset = test_dataset

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=50)
        self.test_dataloader = test_dataloader

    def __get_traindataset(self):
        train_dataset= self.datasetclass(args=args, tokenizer=self.tokenizer, split="train")
        train_dataset.cuda()
        self.train_dataset = train_dataset
        train_batch_size = self.args.per_gpu_train_batch_size*self.args.n_gpu
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        self.train_dataloader = train_dataloader
    def __train(self):
        pass
    def __test_after_train(self):
        pass
    def __get_optimizer_and_criterion(self):
        pass
    def __get_model(self):
        self.model = get_model()
    def train_model(self):
        #获取训练集，模型，opt&criterion
        self.__get_traindataset()
        self.__get_model()

        self.__get_optimizer_and_criterion()
        #训练
        self.__train()
        #测试
        mic, mac = self.__test_after_train()
        pass
if __name__ == "__main__":
    set_seed(args.seed)
    runner = Runner(args)
    runner.train_model()
