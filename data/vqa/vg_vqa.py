import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class VisualGenome_VQA(DatasetLoader):
    """VisualGenomeのVQAデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train", **kwargs):
        super().__init__(**kwargs)        
        tsv_path = os.path.join(data_dir, f"{phase}_qa.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            img_id, question, answer = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, "images_256", f"{img_id}.png")
            self.images.append(img_path)
            self.src_texts.append(question)
            self.tgt_texts.append(answer)