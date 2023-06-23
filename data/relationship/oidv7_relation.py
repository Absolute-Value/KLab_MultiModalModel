import os
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch
from torchvision.transforms import ToTensor
import random


class OpenImageDataset_relation(torch.utils.data.Dataset):
    def __init__(self,data_dir="/data/dataset/openimage",phase="train",imagesize=(256,256),is_mask=True):
        if phase =="val":
            self.phase = "validation"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize
        self.is_mask = is_mask

        rel = pd.read_csv(f"/data/dataset/openimage/relation/oidv6-{self.phase}-annotations-vrd.csv")
        #isは関係が崩壊しているので除外
        self.items = rel[rel['RelationshipLabel']!="is"] 
        self.labels = pd.read_csv("/data/dataset/openimage/oidv7-class-descriptions.csv")
    def _get_location(self,item):
        lb1_x1 = int(self.imagesize[0]*item['XMin1'])
        lb1_x2 = int(self.imagesize[0]*item['XMax1'])
        lb1_y1 = int(self.imagesize[1]*item['YMin1'])
        lb1_y2 = int(self.imagesize[1]*item['YMax1'])
        lb2_x1 = int(self.imagesize[0]*item['XMin2'])
        lb2_x2 = int(self.imagesize[0]*item['XMax2'])
        lb2_y1 = int(self.imagesize[1]*item['YMin2'])
        lb2_y2 = int(self.imagesize[1]*item['YMax2'])
        return (lb1_x1,lb1_y1,lb1_x2,lb1_y2),(lb2_x1,lb2_y1,lb2_x2,lb2_y2)

    def __getitem__(self,idx):
        item= deepcopy(self.items.iloc[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',self.phase,f"{item['ImageID']}.jpg")).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        Label1 = self.labels[self.labels.LabelName==item["LabelName1"]].iloc[0,1]
        Label2 = self.labels[self.labels.LabelName==item["LabelName2"]].iloc[0,1]
        loc1,loc2 = self._get_location(item)
        lb1 = f"\'loc{loc1[0]} loc{loc1[1]} loc{loc1[2]} loc{loc1[3]} {Label1}"
        lb2 = f"\'loc{loc2[0]} loc{loc2[1]} loc{loc2[2]} loc{loc2[3]} {Label2}"
        if self.is_mask:
            src_text = random.choice([f"<extra_id_1> {item['RelationshipLabel']} {Label2}",f"{Label1} {item['RelationshipLabel']} <extra_id_2>",f"<extra_id_1> {item['RelationshipLabel']} <extra_id_2>"])
            tgt_text = f"{Label1} {item['RelationshipLabel']} {Label2}"
        else:
            src_text = f"What is the relationship between {lb1} and {lb2}"
            tgt_text = f"{Label1} {item['RelationshipLabel']} {Label2}"
        return image,src_text,tgt_text

    def get_all(self,idx:int):
        item= deepcopy(self.items.iloc[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',self.phase,f"{item['ImageID']}.jpg")).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        Label1 = self.labels[self.labels.LabelName==item["LabelName1"]].iloc[0,1]
        Label2 = self.labels[self.labels.LabelName==item["LabelName2"]].iloc[0,1]
        loc1,loc2 = self._get_location(item)
        lb1 = f"\'loc{loc1[0]} loc{loc1[1]} loc{loc1[2]} loc{loc1[3]} {Label1}"
        lb2 = f"\'loc{loc2[0]} loc{loc2[1]} loc{loc2[2]} loc{loc2[3]} {Label2}"
        if self.is_mask:
            src_text = random.choice([f"<extra_id_1> {item['RelationshipLabel']} {Label2}",f"{Label1} {item['RelationshipLabel']} <extra_id_2>",f"<extra_id_1> {item['RelationshipLabel']} <extra_id_2>"])
            tgt_text = f"{Label1} {item['RelationshipLabel']} {Label2}"
        else:
            src_text = f"What is the relationship between {lb1} and {lb2}"
            tgt_text = f"{Label1} {item['RelationshipLabel']} {Label2}"
        return image,src_text,tgt_text,loc1,loc2

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    from PIL import ImageDraw
    import torchvision
    index = 2
    dataset = OpenImageDataset_relation(is_mask=True)
    print(dataset.get_all(1))
    # data = dataset.get_all(idx=index)
    # p = torchvision.transforms.functional.to_pil_image(data[0])
    # draw = ImageDraw.Draw(p)
    # draw.rectangle(data[3])
    # p.show()
    # print(f"q:{data[1]}")
    # print(f"a:{data[2]}")
    