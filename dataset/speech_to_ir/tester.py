
"""
2020_09_25
Parent class for IR data handling
"""

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import numpy as np
import soundfile as sf
from scipy import signal

import torch
from torch.utils.data import Dataset, ConcatDataset

import glob
import random
import librosa

from metric.ir_profile import ir_profile
from dataset.rir_dataset.augment_2 import *
from dataset.rir_dataset.pyroom_generator import *



