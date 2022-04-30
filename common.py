import os
import pdb
import torch
import random
import numpy as np
import torch.nn as nn

from tqdm import trange
from torch.optim import AdamW
from pandas_datareader import data as datareader

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)

SEED    = 42 
device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
RD      = lambda x: np.round(x, 3)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

seed_everything(seed = SEED)