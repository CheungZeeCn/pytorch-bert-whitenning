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
from libs import BW, utils
from libs.BW import BWSentenceEmb

bert_model_path = '/home/ana/data1/opdir/fairseq1.0/fairseq/models/bert-base-chinese'


def main():
    texts = ["您好啊", "您好", "今天天气不错"]
    bert_path = '/home/ana/data1/opdir/fairseq1.0/fairseq/models/bert-base-chinese'
    pooling_mode_cls_token = True
    pooling_mode_max_tokens = True
    pooling_mode_mean_tokens = True

    bw = BWSentenceEmb.init_model_from_bert_path(bert_path, pooling_mode_cls_token,
                                                 pooling_mode_max_tokens, pooling_mode_mean_tokens)
    bw.batch_calc_kernel_bias(texts)
    m_path = os.path.join(kp_setup.model_dir, 'BW')
    utils.make_sure_dir_there(m_path)
    bw.store_model_to_path(m_path)
    t1 = bw.texts_sentence_emb("ninhao")
    t2 = bw.texts_sentence_emb("ninhao")
    print(bw.kernel[0][:3], bw.bias[0][:3])
    bw2 = BWSentenceEmb.restore_model_from_path(m_path, bert_path, pooling_mode_cls_token,
                                                pooling_mode_max_tokens, pooling_mode_mean_tokens)
    # bw2.encoder = bw.encoder
    t3 = bw2.texts_sentence_emb("ninhao")
    t4 = bw2.texts_sentence_emb("ninhao")
    print(t1[0][:10])
    print(t2[0][:10])
    print(t3[0][:10])
    print(t4[0][:10])
    print(bw2.kernel[0][:3], bw2.bias[0][:3])


if __name__ == '__main__':
    main()
