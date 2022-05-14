import os
import pdb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

from tqdm import trange
from typing import List, Dict
from torch.optim import AdamW, Adam
from IPython.display import display
from abc import ABC, abstractmethod
from pandas_datareader import data as datareader
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from stockstats import wrap
import pyfolio

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
DEVICE  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
RD      = lambda x: np.round(x, 3)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

STAGE   = 0         # if we change the structure of the GlobalLogger.csv we increase the stage number
USER    = "andreig"

seed_everything(seed = SEED)

class EpsilonScheduler():
    def __init__(self, epsilon = 1.0, epsilon_final = 0.01, epsilon_decay = 0.995):
        self.epsilon       = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def get(self):
        return self.epsilon

    def step(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

class GlobalLogger:
    def __init__(self, path_to_global_logger: str, save_to_log: bool):
        self.save_to_log = save_to_log
        self.path_to_global_logger = path_to_global_logger

        if os.path.exists(self.path_to_global_logger):
            self.logger = pd.read_csv(self.path_to_global_logger)
        else:
            # create folder if not exist
            os.makedirs(os.path.dirname(self.path_to_global_logger), exist_ok=True)

    def append(self, config_file: Dict, output_file: Dict):
        if self.save_to_log == False: return

        if os.path.exists(self.path_to_global_logger) == False:
            config_columns = [key for key in config_file.keys()]
            output_columns = [key for key in output_file.keys()]

            columns = config_columns + output_columns 
            logger = pd.DataFrame(columns = columns)
            logger.to_csv(self.path_to_global_logger, index = False)
            
        self.logger = pd.read_csv(self.path_to_global_logger)
        sample = {**config_file, **output_file}
        columns = [key for (key, value) in sample.items()]

        row = [value for (key, value) in sample.items()]
        row = np.array(row)
        row = np.expand_dims(row, axis = 0)

        sample = pd.DataFrame(row, columns = columns)
        self.logger = self.logger.append(sample, ignore_index = True)
        self.logger.to_csv(self.path_to_global_logger, index = False)

    def get_version_id(self):
        if os.path.exists(self.path_to_global_logger) == False: return 0
        logger = pd.read_csv(self.path_to_global_logger)
        ids = logger["id"].values
        if len(ids) == 0: return 0
        return ids[-1] + 1
    
    def view(self):
        from IPython.display import display
        display(self.logger)


class Logger:
    def __init__(self, path_to_logger: str = 'logger.log', distributed = False):
        from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        if distributed == False:
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler1)

        handler2 = FileHandler(filename = path_to_logger)
        handler2.setFormatter(Formatter("%(message)s"))
        self.logger.addHandler(handler2)

    def print(self, message):
        self.logger.info(message)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
