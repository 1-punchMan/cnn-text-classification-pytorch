# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os, numpy as np, json, pickle, random
import torch

from .dataset import Style_Classification_Dataset
from utils import from_path_import

# from dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD, Dictionary
name = "dictionary"
path = "/home/zchen/encyclopedia-text-style-transfer/dictionary.py"
demands = ["BOS_WORD", "EOS_WORD", "PAD_WORD", "UNK_WORD", "MASK_WORD", "Dictionary"]
from_path_import(name, path, globals(), demands)


logger = getLogger()

def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    dico = data['dico']
    assert ((data['sentences'].dtype == np.uint16) and (len(dico) < 1 << 16) or
            (data['sentences'].dtype == np.int32) and (1 << 16 <= len(dico) < 1 << 31))
    logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data." % (
        len(data['sentences']) - len(data['positions']),
        len(dico), len(data['positions']),
        sum(data['unk_words'].values()), len(data['unk_words']),
        100. * sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions']))
    ))
    # if params.max_vocab != -1:
    #     assert params.max_vocab > 0
    #     logger.info("Selecting %i most frequent words ..." % params.max_vocab)
    #     dico.max_vocab(params.max_vocab)
    #     data['sentences'][data['sentences'] >= params.max_vocab] = dico.index(UNK_WORD)
    #     unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
    #     logger.info("Now %i unknown words covering %.2f%% of the data."
    #                 % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    # if params.min_count > 0:
    #     logger.info("Selecting words with >= %i occurrences ..." % params.min_count)
    #     dico.min_count(params.min_count)
    #     data['sentences'][data['sentences'] >= len(dico)] = dico.index(UNK_WORD)
    #     unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
    #     logger.info("Now %i unknown words covering %.2f%% of the data."
    #                 % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    if (data['sentences'].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data['sentences'] = data['sentences'].astype(np.uint16)

    return data

def set_dico_parameters(params, data, dico):
    """
    Update dictionary parameters.
    """
    if 'dico' in data:
        assert data['dico'] == dico
    else:
        data['dico'] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    if hasattr(params, 'bos_index'):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index

def load_data(params):
    data = {}
    for splt in ['train', 'valid', 'test']:

        # no need to load training data for evaluation
        if splt == 'train' and (params.test or params.predict is not None):
            continue

        # load binarized datasets
        wiki_path = os.path.join(params.wiki_dir, f"{splt}.pth")
        baidu_path = os.path.join(params.baidu_dir, f"{splt}.pth")
        wiki_data = load_binarized(wiki_path, params)
        baidu_data = load_binarized(baidu_path, params)

        # update dictionary parameters
        set_dico_parameters(params, data, wiki_data['dico'])

        # create ParallelDataset
        dataset = Style_Classification_Dataset(wiki_data, baidu_data, splt == 'train', params)

        data[splt] = dataset
        logger.info("")

    id2label = ["wiki", "baidu"]
    data["id2label"] = id2label

    logger.info("")
    return data