from lib2to3.pgen2 import token
import os
import sys
import pickle
import numpy as np
import pandas as pd

# from data.sampler import SubsetSequentialSampler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Huggingface imports
# import datasets
import loralib
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.adapters import CompacterConfig

from transformer_model import HyperTransformer, HyperDistilBertForSequenceClassification

# Global variable
nlp_dataset = None

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (int(self.indices[i]) for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)  

class Logger(object):
    def __init__(self, location):
        self.terminal = sys.stdout
        self.log = open(location, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
        

def set_random_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed(seed+3)
    torch.cuda.manual_seed_all(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_classes(dataset: str) -> int:
    """
    Return the number of classes in the given dataset
    Args:
        dataset: Dataset name (e.g., cifar10, cifar100)
    Returns:
        the number of classes
    """
    n_classes = 0
    if dataset == 'ag_news':
        n_classes = 4
    elif dataset == 'trec':
        n_classes = 6
    elif dataset == 'gpt3mix/sst2':
        n_classes = 2
    elif dataset == 'tweet_eval':
        n_classes = 20
    elif dataset == 'SetFit/20_newsgroups':
        n_classes = 20
    elif dataset == 'multi_sent':
        n_classes = 10
    if not n_classes:
        print('No {} dataset in data directory'.format(dataset))
        exit()
    return n_classes

def get_tokenizer(model: str, max_length: int):
    return AutoTokenizer.from_pretrained(model, max_length=128)

def partition_multi_lang_data(dataset: str, datadir: str, model: str, partition: str, n_parties: int = 10, beta: float = 0.4, max_length: int = 256):
    global nlp_dataset, total_dataset
    raw_dataset = load_dataset('xglue', 'nc')

    multi_train_sets = []
    multi_test_sets = []
    langs = ['en', 'de', 'es', 'fr', 'ru']
    for lang in langs:
        train = raw_dataset['validation.{}'.format(lang)]
        train = pd.DataFrame(train)
        train['language'] = [lang for _ in range(len(train))]
        train['label'] = train['news_category']
        train['text'] = train['news_title'] + ' ' + train['news_body']
        train = Dataset.from_pandas(train).remove_columns(['news_title', 'news_body', 'news_category'])
        
        train = train.train_test_split(test_size=0.2)

        multi_train_sets.append(train['train'])
        multi_test_sets.append(train['test'])

    multi_train_sets = concatenate_datasets(multi_train_sets)
    multi_test_sets = concatenate_datasets(multi_test_sets)
    multi_datasets = DatasetDict({'train': multi_train_sets, 'test':multi_test_sets})

    tokenizer = get_tokenizer(model=model, max_length=max_length)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_dataset = multi_datasets.map(tokenize_function, batched=True)

    # Client Language Groups 
    # 1. Select specific languages
    # 2. Divide clients based on languages
    langs = ['en', 'de', 'es', 'fr', 'ru']
    n_clients_per_language = int(n_parties / len(langs))
    if n_clients_per_language == 0:
        n_clients_per_language = 1

    net_dataidx_map = {}
    for lang in langs:
        df_train = pd.DataFrame(tokenized_dataset['train'])
        df_train = df_train[df_train['language'] == lang]
        y_train = df_train['label'].to_numpy()
        n_train = y_train.shape[0]

        if partition == "noniid-labeldir" or partition == "noniid":
            K = get_num_classes(dataset=dataset)
            
            min_size = 0
            min_require_size = 1
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(n_clients_per_language)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(beta, n_clients_per_language))
                    proportions = np.array([p * (len(idx_j) < n_train / n_clients_per_language) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_clients_per_language):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[len(net_dataidx_map)] = idx_batch[j]

    df_train = pd.DataFrame(tokenized_dataset['train'])
    df_test = pd.DataFrame(tokenized_dataset['test'])
    df_train['id'] = [i for i in range(len(df_train))]
    df_test['id'] = [i for i in range(len(df_test))]

    nlp_dataset = {} # client id -> language // language -> nlp_dataset
    for lang in langs:
        df_lang_train = Dataset.from_pandas(df_train[df_train['language'] == lang]).remove_columns(['language', 'text'])
        df_lang_train.set_format("torch")

        df_lang_test = Dataset.from_pandas(df_test[df_test['language'] == lang]).remove_columns(['language', 'text'])
        df_lang_test.set_format("torch")
        
        nlp_dataset[lang] = [df_lang_train, df_lang_test]
    return net_dataidx_map, None


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))

    print('* Dataset statistics:')
    for net_id in net_cls_counts:
        cd = []
        for c in net_cls_counts[net_id]:
            cd.append(net_cls_counts[net_id][c])
        print('- Client {}: {})'.format(net_id, net_cls_counts[net_id]))
    return net_cls_counts

def get_dataloader(dataset: str, datadir: str, train_bs: int, test_bs: int, dataidxs=None):
    train_dataset = nlp_dataset["train"]
    test_dataset = nlp_dataset["test"]
    if dataidxs is None:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True)
    else:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs), pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetSequentialSampler(dataidxs), pin_memory=True)
    return train_dl, test_dl, local_dl

def get_multi_dataloader(dataset: str, datadir: str, train_bs: int, test_bs: int, dataidxs=None, client_id: int=None, lang=None, n_parties=100):
    langs = ['en', 'de', 'es', 'fr', 'ru']

    if client_id is not None:
        n_clients_per_language = int(n_parties / len(langs))
        if n_clients_per_language == 0:
            n_clients_per_language = 1

        lang_idx = int(client_id/n_clients_per_language)
        current_lang = langs[lang_idx]
        train_dataset, test_dataset = nlp_dataset[current_lang]

    if lang is not None:
        train_dataset, test_dataset = nlp_dataset[lang]

    if n_parties == 1: # Union mode
        all_lang_train_dataset = []
        all_lang_test_dataset = []
        for lang in langs:
            train_dataset, test_dataset = nlp_dataset[lang]
            all_lang_train_dataset.append(train_dataset)
            all_lang_test_dataset.append(test_dataset)
        all_lang_train_dataset = concatenate_datasets(all_lang_train_dataset)
        all_lang_test_dataset = concatenate_datasets(all_lang_test_dataset)
        
        train_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(all_lang_test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True)
        
        return train_dl, test_dl, local_dl
    else:
        if dataidxs is None:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True)
        else:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs), pin_memory=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetSequentialSampler(dataidxs), pin_memory=True)
    return train_dl, test_dl, local_dl

def initialize_networks(dataset: str, model: str, device: str ='cpu', adapter: str = '', rank=16, con_dim=32):
    """ Initialize the network based on the given dataset and model specification. """
    n_classes = get_num_classes(dataset)
    
    plm = AutoModelForSequenceClassification.from_pretrained(model, num_labels=n_classes)

    if adapter == 'hyper':
        base_plm = HyperDistilBertForSequenceClassification.from_pretrained(model, num_labels=n_classes)
        plm = HyperDistilBertForSequenceClassification.from_pretrained(model, num_labels=n_classes)
        plm.distilbert.transformer = HyperTransformer(config=plm.config, rank=rank, con_dim=con_dim)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        
    return plm