# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


logger = getLogger()

class Style_Classification_Dataset(Dataset):

    def __init__(self, wiki_data, baidu_data, train, params):
        self.bos_index = params.bos_index
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        self.sent1, self.sent2 = wiki_data['sentences'], baidu_data['sentences']
        self.pos1, self.pos2 = wiki_data['positions'], baidu_data['positions']
        self.lengths1, self.lengths2 = (
            self.pos1[:, 1] - self.pos1[:, 0],
            self.pos2[:, 1] - self.pos2[:, 0]
        )

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        self.pos1, self.lengths1 = self.remove_empty_sentences(self.pos1, self.lengths1)
        self.pos2, self.lengths2 = self.remove_empty_sentences(self.pos2, self.lengths2)

        self.index = list(range(len(self.pos1))) + list(range(len(self.pos2)))
        self.targets = [0] * len(self.pos1) + [1] * len(self.pos2)
        self.index = list(zip(self.targets, self.index))

        # sanity checks
        self.check()

        self.dataloader = DataLoader(
            self,
            batch_size=params.batch_size,
            shuffle=train,
            drop_last=train,
            collate_fn=self.collate)

    def __getitem__(self, i):
        targ, ind = self.index[i]
        a, b = self.pos2[ind] if targ else self.pos1[ind]
        sent = self.sent2[a:b] if targ else self.sent1[a:b]
        bos, eos = np.array([self.bos_index]), np.array([self.eos_index])
        sent = np.concatenate([bos, sent, eos])
        sent = torch.from_numpy(sent.astype(np.int64))
        return sent, targ

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.index)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
        assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()
        # assert self.lengths.min() > 0                                     # check empty sentences

    def remove_empty_sentences(self, pos, lengths):
        """
        Remove empty sentences.
        """
        init_size = len(pos)
        indices = np.arange(len(pos))
        indices = indices[lengths[indices] > 0]
        pos = pos[indices]
        lengths = pos[:, 1] - pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        return pos, lengths

    def remove_long_sentences(self, pos, lengths, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(pos)
        indices = np.arange(len(pos))
        indices = indices[lengths[indices] <= max_len]
        pos = pos[indices]
        lengths = pos[:, 1] - pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        return pos, lengths

    def collate(self, samples):
        inputs = [s[0] for s in samples]
        inputs = pad_sequence(inputs, padding_value=self.pad_index, batch_first=True)  # (bs, slen)
        
        target = [s[1] for s in samples]
        target = torch.LongTensor(target)

        return inputs, target
        # inputs size: (bs, slen)
        # target size: (bs)