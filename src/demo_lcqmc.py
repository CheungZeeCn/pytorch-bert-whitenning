#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   demo.py
@Contact :   zhangz

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
3/3/2021 12:58 上午   zhangz     1.0         初始版本
"""

import os
import kp_setup
from confs import conf
from libs import BW, utils
from libs.BW import BWSentenceEmb
import pandas as pd
import torch
import logging


def train_one_batch(texts, bert_path, n_layers=2, pooling_mode_cls_token=True, pooling_mode_max_tokens=True,
                    pooling_mode_mean_tokens=True, device=None):
    bw = BWSentenceEmb.init_model_from_bert_path(bert_path, n_layers, pooling_mode_cls_token=pooling_mode_cls_token,
                                                 pooling_mode_max_tokens=pooling_mode_max_tokens,
                                                 pooling_mode_mean_tokens=pooling_mode_mean_tokens, device=device)
    bw.batch_calc_kernel_bias(texts)
    return bw


def train_one_iter(texts, bert_path, n_layers=2, batch_size=128, pooling_mode_cls_token=True,
                   pooling_mode_max_tokens=True,
                   pooling_mode_mean_tokens=True, device=None):
    bw = BWSentenceEmb.init_model_from_bert_path(bert_path, n_layers, pooling_mode_cls_token=pooling_mode_cls_token,
                                                 pooling_mode_max_tokens=pooling_mode_max_tokens,
                                                 pooling_mode_mean_tokens=pooling_mode_mean_tokens, device=device)
    # batch infer
    # all infer
    bw.mini_batch_calc_kernel_bias(texts, batch_size=batch_size, dim=256, max_length=64)
    return bw


def load_tsv_data(fn):
    return pd.read_csv(fn, sep='\t')


def load_all_texts(data_dir):
    fns = ['train.tsv', 'dev.tsv', 'test.tsv']
    texts = []
    for fn in fns:
        df = load_tsv_data(os.path.join(data_dir, fn))
        # df = df[:10000]
        texts += list(df['text_a'])
        texts += list(df['text_b'])
    texts = list(set(texts))
    return texts


def calc_simi(bw, texts1, texts2, max_length=None):
    if max_length is None:
        max_length = max([len(one) for one in texts1+texts2]) + 2
    l = len(texts1)
    with torch.no_grad():
        vecs = bw.texts_sentence_emb(texts1 + texts2, batch_size=256, max_length=max_length)
        # split
        vecs1, vecs2 = vecs[:l], vecs[l:]
        simis = torch.sum(vecs1 * vecs2, dim=-1)
    return simis


def main():
    texts = load_all_texts(os.path.join(kp_setup.data_dir, 'lcqmc'))
    # print(max([len(t) for t in texts]))
    bert_path = conf.bert_model_path
    # bw = train_one(texts, bert_path, pooling_mode_cls_token=False, pooling_mode_max_tokens=False,
    #                pooling_mode_mean_tokens=True)
    df = load_tsv_data(os.path.join(kp_setup.data_dir, 'lcqmc', 'test.tsv'))

    bw = train_one_iter(texts, bert_path, batch_size=256, pooling_mode_cls_token=False, pooling_mode_max_tokens=False, pooling_mode_mean_tokens=True)
    utils.make_sure_dir_there(os.path.join(kp_setup.model_dir, 'BW.001'))
    bw.store_model_to_path(os.path.join(kp_setup.model_dir, 'BW.001'))
    df['BW.001_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']), max_length=64)

    # bw = BWSentenceEmb.restore_model_from_path(os.path.join(kp_setup.model_dir, 'BW.001'), bert_path,
    #                                             pooling_mode_cls_token=False, pooling_mode_max_tokens=False,
    #                                             pooling_mode_mean_tokens=True)
    # df['restore.BW.001_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']), max_length=64)
    # # df['BW.001_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']), max_length=64)
    utils.make_sure_dir_there(kp_setup.output_dir)
    df.to_csv(os.path.join(kp_setup.output_dir, 'lcqmc_test_out.tsv'), sep="\t", index=False)

    # logging.info('BW.001_score DONE')

    # bw = train_one_iter(texts, bert_path, batch_size=128, pooling_mode_cls_token=False, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=False)
    # utils.make_sure_dir_there(os.path.join(kp_setup.model_dir, 'BW.010'))
    # bw.store_model_to_path(os.path.join(kp_setup.model_dir, 'BW.010'))
    # df['BW.010_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']))
    # utils.make_sure_dir_there(kp_setup.output_dir)
    # df.to_csv(os.path.join(kp_setup.output_dir, 'lcqmc_test_out.tsv'), sep="\t", index=False)
    # logging.info('BW.010_score DONE')

    # bw = train_one_iter(texts, bert_path, batch_size=128, pooling_mode_cls_token=False, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=True)
    # utils.make_sure_dir_there(os.path.join(kp_setup.model_dir, 'BW.011'))
    # bw.store_model_to_path(os.path.join(kp_setup.model_dir, 'BW.011'))
    # df['BW.011_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']))
    # utils.make_sure_dir_there(kp_setup.output_dir)
    # df.to_csv(os.path.join(kp_setup.output_dir, 'lcqmc_test_out.tsv'), sep="\t", index=False)
    # logging.info('BW.011_score DONE')

    # bw = train_one_iter(texts, bert_path, batch_size=128, pooling_mode_cls_token=True, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=True)
    # utils.make_sure_dir_there(os.path.join(kp_setup.model_dir, 'BW.111'))
    # bw.store_model_to_path(os.path.join(kp_setup.model_dir, 'BW.111'))
    # df['BW.111_score'] = calc_simi(bw, list(df['text_a']), list(df['text_b']))
    # utils.make_sure_dir_there(kp_setup.output_dir)
    # df.to_csv(os.path.join(kp_setup.output_dir, 'lcqmc_test_out.tsv'), sep="\t", index=False)
    # logging.info('BW.111_score DONE')


if __name__ == '__main__':
    main()
