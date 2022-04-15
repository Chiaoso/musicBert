from turtle import forward
from fairseq.checkpoint_utils import load_pretrained_component_from_model
import fairseq.tasks.sentence_prediction
import fairseq.tasks.masked_lm
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.data import (MaskTokensDataset,
                          LanguagePairDataset,
                          PrependTokenDataset,
                          data_utils)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import TransformerModel
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import math
import logging
import os
import torch

logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256

@register_model("musictrans")
class MusicTransModel(TransformerModel):
    



class OctupleTokenDataset(PrependTokenDataset):
    def adaptor(self, e):
        prev_bar = None
        prev_pos = None
        prev_prog = None
        new_e = []
        for i in e:
            if prev_bar != i[0]:
                prev_bar = i[0]
                prev_pos = None
                new_e.append((i[0], None, None, None, None, None, i[6], None))
            if prev_pos != i[1]:
                prev_pos = i[1]
                prev_prog = None
                new_e.append((None, i[1], None, None, None, None, None, i[7]))
            if prev_prog != i[2]:
                prev_prog = i[2]
                new_e.append((None, None, i[2], None, None, None, None, None))
            if True:
                new_e.append((None, None, None, i[3], i[4], i[5], None, None))
        return new_e

    def convert(self, item):
        encoding = item[8: -8].tolist()
        encoding = list(tuple(encoding[i: i + 8])
                        for i in range(0, len(encoding), 8))
        encoding = self.adaptor(encoding)
        if convert_encoding == 'CP':
            encoding = list(3 if j is None else j for i in encoding for j in i)[
                :crop_length * 8]
        elif convert_encoding == 'REMI':
            encoding = list(j for i in encoding for j in i if j is not None)[
                :crop_length]
        else:
            assert False, 'Unknown encoding format'
        bos = 0
        eos = 2
        encoding = ([bos] * 8) + encoding + ([eos] * 8)
        return torch.tensor(encoding)

    def __init__(self, dataset, token=None):
        super().__init__(dataset, token=None)
        if convert_encoding != 'OCTMIDI':
            self._sizes = np.array([len(self.convert(i)) for i in dataset])
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if convert_encoding != 'OCTMIDI':
            item = self.convert(item)
        return item

    def num_tokens(self, index):
        return self._sizes[index].item()

    def size(self, index):
        return self._sizes[index].item()


fairseq.tasks.sentence_prediction.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.translation.PrependTokenDataset = OctupleTokenDataset
