import os
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch
from torchvision.transforms import ToTensor

#存在しない画像を除外するためのリスト
dropimageidlist = ["7f1934f5884fad79"]

class OpenImageDataset_detection(torch.utils.data.Dataset):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train",imagesize:tuple[int,int]=(256,256)):
        
        if phase=="val":
            self.phase = "validation"
        else:
            self.phase = phase

        self.data_dir = data_dir
        self.imagesize = imagesize
        self.transform = ToTensor()

        datapath = os.path.join(data_dir,"bbox",f"{self.phase}-annotations-bbox.csv")
        self.df = pd.read_csv(datapath)
        #グループの画像を除外する場合はコメントアウコメントを外す
        # self.df = self.df[self.df['IsGroupOf']==0]
        #dropimageidlistに含まれる画像を除外する
        self.df = self.df[self.df["ImageID"].isin(dropimageidlist)==False]
        leabelpath = os.path.join(data_dir,"oidv7-class-descriptions.csv")
        self.labels = pd.read_csv(leabelpath)
        self.imagelist = self.df.ImageID.unique()
    
    def _return_loc(self,imsize:tuple[int,int],bbox: list[float,float,float,float])->tuple[int,int,int,int]:
        """locationを返す

        Parameters
        ----------
        imsize : tuple[int,int]
            画像のサイズ
        bbox : list[float,float,float,float]
            bboxの情報

        Returns
        -------
        x1,x2,y1,y2 : int,int,int,int
            locationのタプル
        """        
        x1 = int(imsize[0]*bbox[0])
        x2 = int(imsize[0]*bbox[2])
        y1 = int(imsize[1]*bbox[1])
        y2 = int(imsize[1]*bbox[3])
        return x1,x2,y1,y2
        
    def __getitem__(self,index):
        imageid = self.imagelist[index]
        df = self.df[self.df.ImageID==imageid]
        imagepath = os.path.join(self.data_dir,self.phase,imageid+".jpg")
        image = self.transform(Image.open(imagepath).convert("RGB").resize(self.imagesize))
        src_text = "What objects are in the image?"
        tgt_text =""
        obj_num = 0
        #画像内のオブジェクト情報からtgt_textを作成
        for _,item in df.iterrows():
            obj_num += 1
            if obj_num > 50:
                break
            Label = self.labels[self.labels.LabelName==item["LabelName"]].iloc[0,1]
            loc = self._return_loc(self.imagesize,item[["XMin","YMin","XMax","YMax"]])
            tgt_text += f'<loc_{loc[0]}> <loc_{loc[2]}> <loc_{loc[1]}> <loc_{loc[3]}> {Label} '
        #tgt_textの最後の空白を削除
        tgt_text = tgt_text.strip()
        return image,src_text,tgt_text
    
    def __len__(self):
        return len(self.imagelist)


if __name__ =="__main__":
    dataset = OpenImageDataset_detection(data_dir="/local_data1/openimage",phase="train")
    for i,data in enumerate(dataset):
        print(i)
    