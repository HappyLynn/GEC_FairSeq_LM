#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('/data/wangzhe/SematicSeg/mlconvgec2018/software/fairseq_py')
from subword_nmt.apply_bpe import *
from fairpy_process import *
from nbest_reranker.features import *
import kenlm
import enchant
import string
if __name__ == '__main__':
    MODEL_DIR = '/data/wangzhe/SematicSeg/mlconvgec2018/models/'
    bpe_model_name = MODEL_DIR + "bpe_model/train.bpe.model"
    inputs_file_name = '/data/wangzhe/SematicSeg/mlconvgec2018/inputs'
    merges = -1
    separator = '@@'
    vocabulary = None
    glossaries = None
    model_dir = '/data/wangzhe/SematicSeg/mlconvgec2018/models/mlconv_embed'
    model_names = glob.glob(model_dir + '/*')
    bpe = BPE(codecs.open(bpe_model_name, encoding='utf-8'), merges, separator, vocabulary, glossaries)
    spell_checker = enchant.Dict("en_US")
    ed = EditOps(name='EditOps0')
    lm = LM('LM0', '/data/wangzhe/SematicSeg/mlconvgec2018/models/lm/94Bcclm.trie', normalize=False)
    wp = WordPenalty(name='WordPenalty0')
    weights = [0.94064, -0.0208803, -0.00450021, 0.015532, 0.00618153, -0.0122658]
    models, generator, align_dict, max_positions, args, use_cuda, task, src_dict, tgt_dict = load(model_names=model_names, use_cpu=False)
    input_sen = "His is plays him ."
    words = input_sen.split()
    totals = []
    candidate_words = 3
    delset = string.punctuation
    for each_word in words:
        if each_word in delset:
            totals.append([each_word])
            continue
        if spell_checker.check(each_word):
            totals.append([each_word])
        else:
            totals.append(spell_checker.suggest(each_word)[:candidate_words])
    cur = []
    prev = [""]
    for i in range(len(totals)):
        for item in prev:
            for j in range(len(totals[i])):
                cur.append((item + ' ' + totals[i][j]).strip())
        prev = cur
        cur = []

    outputs, ori_scores = model_predict(prev, models, generator, align_dict, max_positions, args, use_cuda, task, src_dict, tgt_dict)
    score_dict = dict()
    for ind, output in enumerate(outputs):
        s0 = ori_scores[ind]
        s1 = [float(item) for item in ed.get_score(input_sen, output).split()]
        s2 = float(lm.get_score(input_sen, output))
        s3 = float(wp.get_score(input_sen, output))
        final_score = s0 * weights[0] + s1[0] * weights[1] + s1[1] * weights[2] + s1[2] * weights[3] + s2 * weights[4] + s2 * weights[5]
        score_dict[ind] = final_score
        print(s0, s1[0], s1[1], s1[2], s2, s3)
    sorted_indices = sorted(score_dict, key=score_dict.get, reverse=True)
    for ind in sorted_indices:
        print(outputs[ind])
    m = 0

