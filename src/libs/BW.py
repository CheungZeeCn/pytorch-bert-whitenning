#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BW.py    
@Contact :   zhangz

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
3/3/2021 12:59 上午   zhangz     1.0         None
"""

from transformers import BertModel, BertTokenizer, BertConfig
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

tokenizer = None
config = None
bert_model = None
encoder = None


def load_bert_by_path(model_path):
    global tokenizer, bert_model, config
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel(config)
    return bert_model, config, tokenizer


def build_model(model_path, n_layers=2,
                pooling_mode_cls_token: bool = False,
                pooling_mode_max_tokens: bool = False,
                pooling_mode_mean_tokens: bool = True):
    bert_model, config, tokenizer = load_bert_by_path(model_path)
    # 加一个池化给它
    # 然后返回
    word_embedding_dimension = config.hidden_size
    pooling = Pooling(word_embedding_dimension=word_embedding_dimension, n_layers=n_layers,
                      pooling_mode_cls_token=pooling_mode_cls_token, pooling_mode_max_tokens=pooling_mode_max_tokens,
                      pooling_mode_mean_tokens=pooling_mode_mean_tokens)
    encoder = Encoder(bert_model, pooling)
    return tokenizer, encoder


def collate_fn_with_tokenizer(tokenizer, sub_texts, max_length):
    t = tokenizer(sub_texts, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
    # t = tokenizer(sub_texts, return_tensors="pt", padding=True, max_length=max_length, truncation=True)
    return t


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 n_layers: int = 2,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True
                 ):
        super(Pooling, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token',
                            'pooling_mode_mean_tokens', 'pooling_mode_max_tokens',
                            'n_layers']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.n_layers = n_layers

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        # token_embeddings = features['last_hidden_state']
        hidden_states = features['hidden_states']
        cls_token = features['pooler_output']
        input_mask = features['attention_mask']

        # print("TEST!!", self.pooling_mode_cls_token, self.pooling_mode_max_tokens, self.pooling_mode_mean_tokens)
        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            output_vectors_max = []
            for i in range(self.n_layers):
                # i_layer = hidden_states[-1 - i]
                # 默认第0层和最后一层 万万没想到 这种骚操作
                i_layer = hidden_states[-i]
                input_mask_expanded = input_mask.unsqueeze(-1).expand(i_layer.size()).float()
                i_layer[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                max_over_time = torch.max(i_layer, 1)[0]
                output_vectors_max.append(max_over_time)
            output_vectors.append(torch.mean(torch.stack(output_vectors_max, -1), -1))
            pass
        if self.pooling_mode_mean_tokens:
            output_vectors_mean = []
            for i in range(self.n_layers):
                # i_layer = hidden_states[-1 - i]
                i_layer = hidden_states[-i]
                input_mask_expanded = input_mask.unsqueeze(-1).expand(i_layer.size()).float()
                sum_embeddings = torch.sum(i_layer * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                if self.pooling_mode_mean_tokens:
                    output_vectors_mean.append(sum_embeddings / sum_mask)
            output_vectors.append(torch.mean(torch.stack(output_vectors_mean, -1), -1))
            pass

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return Pooling(**config)


class Encoder(nn.Module):
    def __init__(self, bert_model, pooling):
        super(Encoder, self).__init__()
        self.bert_model = bert_model
        self.pooling = pooling
        self.emb_dim = pooling.word_embedding_dimension

    def forward(self, x):
        # print(x.keys())
        y = self.bert_model(**x, output_hidden_states=True, return_dict=True)
        x.update(y)
        x = self.pooling(x)
        return x['sentence_embedding']


# class BWSentenceEmb(object):
class BWSentenceEmb(nn.Module):
    def __init__(self, tokenizer, encoder, kernel=None, bias=None, device=None):
        # super(BWSentenceEmb).__init__()
        super(BWSentenceEmb, self).__init__()
        self.encoder = encoder
        self.kernel = kernel
        self.bias = bias
        self.tokenizer = tokenizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        self.to(device)
        self.encoder.to(device)
        self.eval()
        #logging.info("encoder device: {}".format(self.encoder.device))


    def forward(self, x):
        x = self.encoder(x)
        x = (x + self.bias).matmul(self.kernel)
        x = x.div(x.norm(p=2, dim=1, keepdim=True))
        return x

    def tokens_sentence_emb(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            x = self.forward(x).detach()
        return x

    def texts_sentence_emb(self, x, batch_size=None, max_length=None):
        self.eval()
        if max_length is None:
            max_length = max([len(one) for one in x]) + 2
        if batch_size is None:
            with torch.no_grad():
                x = self.tokenizer(x, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
                x = x.to(self.device)
                x = self.forward(x)
        else:
            dataloader = DataLoader(x, batch_size=batch_size,
                                    collate_fn=lambda sub_x: collate_fn_with_tokenizer(self.tokenizer, sub_x, max_length),
                                    sampler=None,
                                    shuffle=False,
                                    num_workers=0
                                    )
            texts_length = len(x)
            logging.info("IN mini_batch_calc_kernel_bias, nums of texts:{}, batch_size:{}".format(texts_length, batch_size))
            # cpu numpy
            mini_batch_vecs_tensors = []
            with torch.no_grad():
                for _, batch in enumerate(tqdm(dataloader)):
                    batch = batch.to(self.device)
                    vecs = self.forward(batch)
                    vecs = vecs.to('cpu')
                    mini_batch_vecs_tensors.append(vecs)
                # stack numpy vecs
                x = torch.cat(mini_batch_vecs_tensors, dim=0).detach()
        return x

    def mini_batch_calc_kernel_bias(self, texts, batch_size=8, dim=None, max_length=None):
        self.eval()
        if max_length is None:
            max_length = max([len(one) for one in texts]) + 2
        dataloader = DataLoader(texts, batch_size=batch_size,
                                collate_fn=lambda x: collate_fn_with_tokenizer(self.tokenizer, x, max_length),
                                sampler=None,
                                shuffle=False,
                                num_workers=0
                                )
        # cpu numpy
        texts_length = len(texts)
        logging.info("IN mini_batch_calc_kernel_bias, nums of texts:{}, batch_size:{}".format(texts_length, batch_size))
        mini_batch_vecs = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                batch = batch.to(self.device)
                i_vecs = self.encoder(batch)
                mini_batch_vecs.append(i_vecs.cpu().detach().numpy())
            # stack numpy vecs
            vecs = np.vstack(mini_batch_vecs)
            mu = vecs.mean(axis=0, keepdims=True)
            cov = np.cov(vecs.T)
            u, s, vh = np.linalg.svd(cov)
            W = np.dot(u, np.diag(1 / np.sqrt(s)))
            if dim is None:
                kernel = Tensor(W).to(self.device)
                bias = Tensor(-mu).to(self.device)
            else:
                kernel = Tensor(W[:, :dim]).to(self.device)
                bias = Tensor(-mu).to(self.device)
        # kernel.requires_grad()
        self.kernel = kernel.detach()
        self.bias = bias.detach()
        return kernel, bias


    def batch_calc_kernel_bias(self, texts, dim=None, max_length=None):
        self.eval()
        if max_length is None:
            t = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True)
        else:
            t = self.tokenizer(texts, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
        # t = self.tokenizer(texts, return_tensors="pt", padding=True)
        t = t.to(self.device)
        with torch.no_grad():
            vecs = self.encoder(t)
            mu = vecs.mean(axis=0, keepdims=True)
            cov = np.cov(vecs.cpu().detach().numpy().T)
            u, s, vh = np.linalg.svd(cov)
            W = np.dot(u, np.diag(1 / np.sqrt(s)))
            # return None, None
            if dim is None:
                kernel = Tensor(W).to(self.device)
                bias = -mu
            else:
                kernel = Tensor(W[:, :dim]).to(self.device)
                bias = -mu
            # kernel.requires_grad()
        self.kernel = kernel.detach()
        self.bias = bias.detach()
        return kernel, bias

    def iter_calc_kernel_bias(self, texts, dim=None):
        max_length = max(len(t) for t in texts)
        print(max_length)
        raise NotImplementedError()

    def store_model_to_path(self, model_path):
        file_name = 'mu_bias.pkl'
        file_name2 = 'model.bin'
        file_loc = os.path.join(model_path, file_name)
        file_loc2 = os.path.join(model_path, file_name2)
        logging.info("storing model into [{}]".format(file_loc))
        with open(file_loc, 'wb') as f:
            pickle.dump((self.kernel, self.bias), f)
        #torch.save((self.kernel, self.bias), file_loc)
        torch.save(self, file_loc2)
        return True

    @classmethod
    def restore_model_from_path(cls, model_path, bert_path, n_layers=2,
                                pooling_mode_cls_token: bool = False,
                                pooling_mode_max_tokens: bool = False,
                                pooling_mode_mean_tokens: bool = True,
                                device=None
                                ):
        # load bert from pretrianed
        tokenizer, encoder = build_model(bert_path, n_layers, pooling_mode_cls_token, pooling_mode_max_tokens,
                                         pooling_mode_mean_tokens)
        #encoder.eval()
        # load model from model_path
        file_name = 'mu_bias.pkl'
        file_name2 = 'model.bin'
        file_loc = os.path.join(model_path, file_name)
        file_loc2 = os.path.join(model_path, file_name2)
        if not os.path.exists(file_loc):
            logging.info("PATH {} not exist, please check it".format(file_loc))
            return None
        with open(file_loc, 'rb') as f:
            kernel, bias = pickle.load(f)
        m = torch.load(file_loc2)
        # kernel = kernel.detach()
        #bias = bias.detach()
        #m = BWSentenceEmb(tokenizer, encoder, kernel, bias, device=device)
        #m.eval()

        return m

    @classmethod
    def init_model_from_bert_path(cls, bert_path, n_layers=2,
                                  pooling_mode_cls_token: bool = False,
                                  pooling_mode_max_tokens: bool = False,
                                  pooling_mode_mean_tokens: bool = True,
                                  device=None
                                  ):
        tokenizer, encoder = build_model(bert_path, n_layers, pooling_mode_cls_token, pooling_mode_max_tokens,
                                         pooling_mode_mean_tokens)

        # encoder.eval()
        # emb_dim = encoder.emb_dim
        # kernel, bias = torch.rand(emb_dim, output_dim), torch.rand(1, emb_dim)
        m = BWSentenceEmb(tokenizer, encoder, device=device)
        m.eval()
        return m


if __name__ == '__main__':
    texts = ["您好啊", "您好", "今天天气不错"]
    bert_path = '/home/ana/data1/opdir/fairseq1.0/fairseq/models/bert-base-chinese'
    model_path = "."
    # tokenizer, encoder = build_model('/home/ana/data1/opdir/fairseq1.0/fairseq/models/bert-base-chinese', 2)
    # t = tokenizer(texts, return_tensors="pt", padding=True)
    # r = encoder(t)
    # kernel, bias = batch_calc_kernel_bias(encoder, tokenizer, texts, 256)
    bw = BWSentenceEmb.init_model_from_bert_path(bert_path)
    # bw.batch_calc_kernel_bias(texts)
    # bw.store_model_to_path("./")
