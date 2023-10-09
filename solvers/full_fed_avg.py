import os
import copy
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import get_num_classes, initialize_networks, get_dataloader, get_multi_dataloader
from torch.cuda.amp import GradScaler 

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

class FullFedAvgSolver(object):
    """ Base class for training and testing federated learning (Research purpose only) """

    def __init__(self, args, client2data: dict, cls_dist=None):
        """ Initialize configurations. """
        self.args = args
        self.client2data = client2data
        self.num_class = get_num_classes(args.dataset)
        self.data_lens = [len(client2data[client_id]) for client_id in range(args.n_parties)]

        # Load training networks
        self.global_net = initialize_networks(dataset=args.dataset, model=args.model, adapter=args.adapter, rank=args.rank, con_dim=args.con_dim)
        print(self.global_net)

        # Load Local data loader
        self.local_train_loader = {}
        for client_id in self.client2data:
            dataidx = self.client2data[client_id]
            loader, _, _ = get_multi_dataloader(dataset=self.args.dataset,
                                        datadir=self.args.datadir,
                                        train_bs=self.args.batch_size,
                                        test_bs=self.args.batch_size,
                                        dataidxs=dataidx, 
                                        client_id=client_id, n_parties=self.args.n_parties)
            self.local_train_loader[client_id] = loader

        # Optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.device = torch.device('cuda')

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
                    
                    outputs = net(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Summary
                    writer['acc'] += torch.eq(logits.argmax(dim=-1), labels).float().mean()
                    writer['step'] += 1
            total_acc += float(writer['acc'] / writer['step'])
            print('lang {}: {}'.format(lang, float(writer['acc'] / writer['step'])))
        return total_acc / len(langs)

    def training_plm(self, client_id: int, networks, test_loader=None, warmup=True):
        """ Perform local optimization (Training global model on the local data """
        # Load the optimizer
        optimizer = torch.optim.AdamW(networks.parameters(), lr=self.args.lr)    
        criterion = nn.CrossEntropyLoss(reduce=False)
        scaler = GradScaler()

        # print('* Training Client ({}/{}) on {} data'.format(client_id + 1, self.args.n_parties, len(self.local_train_loader[client_id]) * self.args.batch_size))
        networks.to(self.device)
        # self.global_net.to(self.device).eval()
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0}
            networks.train()
            for batch in self.local_train_loader[client_id]:
                indexes = batch['id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
        
                outputs = networks(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Model Updates
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
        
    def aggregate(self, local_nets: dict, cohorts: list):
        """ Create global model based on the trained local models (Clients) """

        data_sum = 0
        for client_id in cohorts:
            data_sum += self.data_lens[client_id]

        self.global_net = self.global_net.cpu()
        g_params = self.global_net.state_dict()
        for key in self.global_net.state_dict():
            for i, client_id in enumerate(cohorts):  # num_batches_tracked
                w = self.data_lens[client_id] / data_sum
                p = local_nets[client_id].state_dict()[key].cpu()
                if i == 0:
                    g_params[key] = (w * p).type(self.global_net.state_dict()[key].dtype)
                else:
                    g_params[key] += (w * p).type(self.global_net.state_dict()[key].dtype)
        self.global_net.load_state_dict(g_params)

    def run(self):
        """ Start federated learning scenario """

        # Start communication rounds
        for round in range(self.args.comm_round):
            cohorts = np.random.choice(self.args.n_parties, int(self.args.n_parties * self.args.sample_fraction), replace=False).tolist()
            local_nets = {}
            for cidx, client_id in enumerate(cohorts):
                local_nets[client_id] = copy.deepcopy(self.global_net)
                self.training_plm(client_id=client_id, networks=local_nets[client_id])
            self.aggregate(local_nets, cohorts)
            test_acc = self.testing_multi_lang_plm()
            print('Round ({}/{}) Test accuracy {}'.format(round + 1, self.args.comm_round, test_acc))
            if (round + 1) % 10 == 0:
                self.save_model(round=round + 1)

    def load_model(self, round):
        file_name = '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.args.model, self.args.beta, self.args.n_parties, self.args.alg, self.args.adapter, self.args.sample_fraction, round)
        self.global_net = pickle.load(open(os.path.join(self.args.modeldir, file_name), 'rb'))

    def save_model(self, round):
        file_name = '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.args.model, self.args.beta, self.args.n_parties, self.args.alg, self.args.adapter, self.args.sample_fraction, round)
        pickle.dump(self.global_net, open(os.path.join(self.args.modeldir, file_name), 'wb'))
