import os
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader, DETECTION_SRC_TEXT

#存在しない画像を除外するためのリスト
dropimageidlist =["7f1934f5884fad79","429019e83c1c2c94","4f818c006da84c9e","5b86e93f8654118a","673d74b7d39741c3","6dcd3ce37a17f2be","805baf9650a12710"
                   ,"98ac2996fc46b56d","a46a248a39f2d97c","9316d4095eab6d10","9ee38bb2e69da0ac","37625d59d0e0782a"]

class OpenImage_Detection(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train"):
        super().__init__()
        if phase=="val":
            phase = "validation"

        with open(os.path.join(data_dir,"tsv_fix",f"{phase}_dec_cut_max_tokens.tsv")) as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        items = [item for item in items if len(item)==2]
        self.tgt_texts = [item[1] for item in items]
        self.src_texts = [DETECTION_SRC_TEXT]*len(items)
        self.images = [os.path.join(data_dir,f"{phase}_256_png",f"{item[0]}.png") for item in items]

        #dropimageidlistに含まれる画像と対応するテキストを除外する
        for drop_id in dropimageidlist:
            drop_path = os.path.join(data_dir,f"{phase}_256_png",f"{drop_id}.png")
            while drop_path in self.images:
                drop_index = self.images.index(drop_path)
                self.tgt_texts.pop(drop_index)
                self.src_texts.pop(drop_index)
                self.images.pop(drop_index)
    
