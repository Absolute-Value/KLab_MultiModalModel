import os
import json
import re
import random
from copy import deepcopy
from PIL import Image
import torch
from torchvision.transforms import ToTensor,functional
import time
import tqdm
import torch.distributed as dist


class Vqa2dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,split="train",imagesize=(256,256)):
        self.split = split
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        self.quetions = json.load(open(f'{self.data_dir}/v2_OpenEnded_mscoco_{self.split}2014_questions.json'))
        self.answers = json.load(open(f'{self.data_dir}/v2_mscoco_{self.split}2014_annotations.json'))
        
    def __getitem__(self,idx):
        src_text = self.quetions['questions'][idx]["question"]
        tgt_text = self.answers['annotations'][idx]["multiple_choice_answer"]
        imgbase='%s/%s/COCO_%s_%012d.jpg'
        imgid = self.quetions['questions'][idx]["image_id"]
        imgpath = imgbase%(self.data_dir,f"{self.split}2014", f"{self.split}2014", imgid)
        image = Image.open(imgpath).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        return image,src_text,tgt_text

    def __len__(self):
        return len(self.items)

def get_vcr_dataloader(args, phase, rank):
    dataset = Vqa2dataset(args.data_dir, split=phase)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count()//4, pin_memory=True, sampler=sampler)
    return dataloader

if __name__ =="__main__":
    _DATADIR = "/data/dataset/vqa2"
    dataset = Vqa2dataset(_DATADIR)
    data = dataset[10]

    print(data)