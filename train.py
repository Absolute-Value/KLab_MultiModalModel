import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import *
from models.model import MyModel

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    args = parse_arguments()
    if rank == 0: os.makedirs(args.result_dir, exist_ok=True)

    # create local model
    model = MyModel(args).to(device_id)
    # construct DDP model
    model = DDP(model, device_ids=[device_id])
    # define optimizer
    optimizer = torch.optim.Adam(model.module.transformer.parameters(), lr=args.lr)

    image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=512)

    # データローダーの設定
    train_dataset = DatasetLoader(args, phase="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_dataset = DatasetLoader(args, phase="val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    min_val_loss = 100
    train_losses = []
    val_losses = []
    for epoch in range(args.num_epochs):
        # 学習ループ
        total_train_loss = 0.0
        model.module.transformer.train()
        train_loop = tqdm(train_loader, desc=f'Train (Epoch {epoch+1}/{args.num_epochs})', disable=(rank != 0))
        for images, src_texts, tgt_texts in train_loop:
            images = image_processor(images, return_tensors="pt").to(device_id)
            source_encoding = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
            target_encoding = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
            loss = model(images, source_encoding, target_encoding)

            # パラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        if rank == 0:
            train_losses.append(total_train_loss/len(train_loader))
            print(f'[Epoch ({epoch+1}/{args.num_epochs})] Train loss : {train_losses[-1]}')

        # 検証ループ
        total_val_loss = 0.0
        model.module.transformer.eval()
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch+1}/{args.num_epochs})', disable=(rank != 0))
        for images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                images = image_processor(images, return_tensors="pt").to(device_id)
                source_encoding = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
                target_encoding = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
                loss = model(images, source_encoding, target_encoding)
                total_val_loss += loss.item()

        if rank == 0:
            val_losses.append(total_val_loss/len(val_loader))
            print(f'[Epoch ({epoch+1}/{args.num_epochs})] Val loss : {val_losses[-1]}')
        
            if val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                model.module.save(args.result_dir)
                print('Model saved')
            
    if rank == 0:
        # Plot the loss values.
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')

        # Set the title and axis labels.
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Show the plot.
        plt.savefig(f"{args.result_dir}/val_loss.png")

if __name__=="__main__":
    train()