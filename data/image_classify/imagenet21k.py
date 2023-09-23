import os
from ..dataset_loader import DatasetLoader

class ImageNet21k_DatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data01/imagenet21k', phase='train'):
        super().__init__()
        with open(os.path.join(data_dir,f"text_{phase}_256fix.tsv"), 'r') as f:
            data = f.read().split('\n')
        with open(os.path.join(data_dir,f"img_{phase}_256fix.tsv"), 'r') as f:
            temp = f.read().split('\n')
        
        # 
        data.extend(temp[1:])
        data = data[1:]
        data = [d.split('\t') for d in data]

        self.images = [os.path.join(data_dir, d[0]) for d in data]
        self.src_texts = ["What does the image describe ?"]*len(data)
        self.tgt_texts = [f'a photo of {d[1]}' for d in data]


        