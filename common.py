import os
import pdb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import trange
from torch.optim import AdamW
from IPython.display import display
from pandas_datareader import data as datareader
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

PATH_TO_DATA = 'data/AAPL_stocks_splits.csv'

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