import os
import torch
import random
import numpy as np


def set_seed(seed: int):
    """ Set random seed for python, numpy and pytorch RNGs """
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
