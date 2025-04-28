import collections
import copy
import json
import os
import sys
from functools import lru_cache
import torch
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from termcolor import colored
import torch
from torch.utils.data import Dataset
import random
import pandas as pd


class BaseDataset(Dataset):
    def __init__(self, data_x : np.ndarray, data_y : List[int], composer = None, infer : bool = False) -> None:
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.composer = composer
        self.infer = infer

    def init_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.data_x.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        index = index % self.data_x.shape[0]

        y = self.data_y[index]
        x = self.data_x[index]

        
        if self.composer is not None:
            x = self.composer(x)

        # x = np.gradient(x, np.arange(x.shape[0]), axis=0)

        x = x.astype(dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.infer:
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

        return x, y


