import os
import copy
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from wandb import visualize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blocks import HyperNetworkTransformerBlock
from utils import get_num_classes, initialize_networks, get_multi_dataloader

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

from solvers.full_fed_avg import FullFedAvgSolver

from torch.cuda.amp import GradScaler 

class HyperFedAvgSolver(FullFedAvgSolver):
    def __init__(self, args, client2data: dict, cls_dist=None):
        super().__init__(args, client2data, cls_dist)

    def testing_multi_lang_plm(self, model=None):
        writer = {'loss': 0., 'acc': 0., 'step': 0}
        if model is None:
            model = self.global_net
        net = model.to(self.device)
        net.eval()

        total_acc = 0.
        langs = ['en', 'de', 'es', 'fr', 'ru']
        for lang in langs:
            writer['loss'] = 0.
            writer['acc'] = 0.
            writer['step'] = 0.
            writer['avg_code'] = 0.
            _, global_loader, _ = get_multi_dataloader(dataset=self.args.dataset,
                                                        datadir=self.args.datadir,
                                                        train_bs=self.args.batch_size,
                                                        test_bs=self.args.batch_size, 
                                                        lang=lang,
                                                        n_parties=self.args.n_parties)
            
            with torch.no_grad():
                for batch in global_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    indexes = batch['id'].to(self.device)
                    labels = batch['label'].to(self.device)

                    batch_statistic = torch.ones(1, self.num_class).to(self.device).float()
                    batch_statistic = batch_statistic / batch_statistic.sum(dim=-1, keepdim=True)
                    net.distilbert.transformer.hyper_nets.dist_stats = batch_statistic
                    
                    outputs = net(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Summary
                    writer['acc'] += torch.eq(logits.argmax(dim=-1), labels).float().sum()
                    writer['step'] += len(logits)
            total_acc += float(writer['acc'] / writer['step'])
            print('lang {}: {}'.format(lang, float(writer['acc'] / writer['step'])))
        return total_acc / len(langs)

    def training_plm(self, client_id: int, networks, test_loader=None, warmup=True, class_tokens=None):
        """ Perform local optimization (Training global model on the local data """
        # Load the optimizer
        optimizer = torch.optim.AdamW(networks.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss(reduce=False)
        scaler = GradScaler()

        # print('* Training Client ({}/{}) on {} data'.format(client_id + 1, self.args.n_parties, len(self.local_train_loader[client_id]) * self.args.batch_size)

        block_book = {}
        for i, (name, module) in enumerate(networks.named_modules()):
            if isinstance(module, HyperNetworkTransformerBlock):
                block_book[len(block_book)] = module

        data_labels = []
        for batch in self.local_train_loader[client_id]:
            labels = batch['label'].to(self.device)
            data_labels.append(labels)
        data_labels = torch.cat(data_labels, dim=0)
        data_labels = F.one_hot(data_labels, self.num_class).float().to(self.device).sum(dim=0, keepdim=True)
        data_labels = data_labels / data_labels.sum(dim=-1, keepdim=True)

        networks.to(self.device)
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0}
            networks.train()
            for name, param in networks.named_parameters():
                if name.startswith('classifier') or name.startswith('pre_classifier') or 'hyper' in name: # choose whatever you like here
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for batch in self.local_train_loader[client_id]:
                indexes = batch['id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                batch_statistic = F.one_hot(labels, self.num_class).float().to(self.device).sum(dim=0, keepdim=True)
                batch_statistic = batch_statistic / batch_statistic.sum(dim=-1, keepdim=True)
                networks.distilbert.transformer.hyper_nets.dist_stats = batch_statistic

                outputs = networks(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = (-F.one_hot(labels, self.num_class) * logits.log_softmax(dim=-1)).sum(dim=-1)
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                writer['loss']  += loss.mean().item()
                writer['acc']   += torch.eq(logits.argmax(dim=-1), labels).float().mean()
                writer['step']  += 1
                
            if self.args.n_parties == 1: # Union
                test_acc = self.testing_multi_lang_plm(model=networks)
                print('Epoch ({}/{}) Test accuracy {}'.format(epoch + 1, self.args.epochs, test_acc))
        networks = networks.cpu()

    def run(self):
        """ Start federated learning scenario """
        # Start communication rounds
        test_acc_hist = []
        for round in range(self.args.comm_round):
            cohorts = np.random.choice(self.args.n_parties, int(self.args.n_parties * self.args.sample_fraction), replace=False).tolist()
            local_nets = {}
            for cidx, client_id in enumerate(cohorts):
                local_nets[client_id] = copy.deepcopy(self.global_net)
                self.training_plm(client_id=client_id, networks=local_nets[client_id])
            self.aggregate(local_nets, cohorts)
            test_acc = self.testing_multi_lang_plm()
            print('Round ({}/{}) Test accuracy {}'.format(round + 1, self.args.comm_round, test_acc))
            test_acc_hist.append(float(test_acc))
            if (round + 1) % 10 == 0:
                print(test_acc_hist)
                self.save_model(round=round + 1)
