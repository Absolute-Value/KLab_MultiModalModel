import os
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ..dataset_loader import DatasetLoader


class Grit20M_RefExp(DatasetLoader):
    """Grit20mのReferring Expressionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/grit20m/",phase:str="train"):
        super().__init__()
        tsv_path = os.path.join(data_dir,  f"{phase}_ref_exp.tsv")
            
        with open(tsv_path) as f:
            items = f.read()
        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        self.tgt_texts = [item[2] for item in items]
        self.src_texts = [item[1] for item in items]
        self.images = [os.path.join(data_dir,item[0]) for item in items]

    
