
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
from fairseq.models.roberta import TransformerSentenceEncoder, RobertaEncoder, RobertaModel
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


from musicbert import MusicBERTEncoder

