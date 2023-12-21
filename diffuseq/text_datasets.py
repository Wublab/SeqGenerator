# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import csv

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
from diffuseq.esm2 import data
from typing import List, Dict, Any
import random
from torch.utils.data.distributed import DistributedSampler



PADDED_FEATS = [
    'input_ids'
]
PADDED_MASKS = [
    'input_mask', 'loss_mask'
]


def pad_feats(raw_feats, max_len):
    origin_len = raw_feats['input_ids'].shape[0]
    padded = [1] * (max_len - origin_len)
    padded = np.array(padded, dtype=np.int64)
    for feat_name in PADDED_FEATS:
        raw_feats[feat_name] = np.concatenate((raw_feats[feat_name], padded))
    padded = [0] * (max_len - origin_len)
    padded = np.array(padded, dtype=np.int64)
    for feat_name in PADDED_MASKS:
        raw_feats[feat_name] = np.concatenate((raw_feats[feat_name], padded))

    return raw_feats

def length_batching(
        np_dicts
    ):
    get_len = lambda x: x['input_ids'].shape[0]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [
        pad_example(x) for (_, x) in length_sorted]
    return torch.utils.data.default_collate(padded_batch)

def load_data_text(
    batch_size, 
    model_emb,
    deterministic=False, 
    data_args=None,
    split='train', 
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, split=split)

    dataset = TextDataset(
        data_args,
        split,
        training_data,
        model_emb
    )
    collate_fn = lambda x: length_batching(x)
    #sampler = DistributedSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        shuffle=not deterministic,
        collate_fn=collate_fn,
        num_workers=0
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, split='train'):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.csv'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.csv'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.csv'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[2],int(r[3])] for r in reader]

    train_dataset = {}
    for r in rows:
        #for generate use 300-340
        #if r[2] >=300 and r[2] <= 375:
        if r[2] <= data_args.max_len  and r[2] >=data_args.min_len:
            if r[0] in train_dataset:
                train_dataset[r[0]].append(r[1])
            else:
                train_dataset[r[0]] = [r[1]]

    return train_dataset



class TextDataset(Dataset):
    def __init__(self, data_args, split, training_data, model_emb):
        super().__init__()
        self.data_dict = training_data

        self.cluster = list(training_data.keys())
        random.shuffle(self.cluster)
        

        alphabet = data.Alphabet.from_architecture("ESM-1b")
        self.converter = alphabet.get_single_converter()
        self.model_emb = model_emb


    def __len__(self):
        return len(self.cluster)

    def __getitem__(self, idx):
        cluster = self.cluster[idx]
        cluster_idx = np.random.randint(0, len(self.data_dict[cluster]))
        seq_str = self.data_dict[cluster][cluster_idx]
        out_kwargs = {}
        out_kwargs['input_ids'] = np.array(self.converter(seq_str))
        out_kwargs['input_mask'] = np.ones(out_kwargs['input_ids'].shape, dtype=out_kwargs['input_ids'].dtype)
        out_kwargs['loss_mask'] = np.ones(out_kwargs['input_ids'].shape, dtype=out_kwargs['input_ids'].dtype)
        mask = np.where(out_kwargs['input_ids'] >= 24)
        out_kwargs['loss_mask'][mask] = 0
        mask = np.where(out_kwargs['input_ids'] == 1)
        out_kwargs['loss_mask'][mask] = 0
        mask = np.where(out_kwargs['input_ids'] == 3)
        out_kwargs['loss_mask'][mask] = 0

        return out_kwargs
        


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result
