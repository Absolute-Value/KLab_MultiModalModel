batch_size=128
dataset="sun397"

epoch=50

enc=2
dec=2
lr=1e-5
torchrun --nnodes=1 --nproc_per_node=1 train.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase train \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LambdaLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /local/ \
        --result_dir results/train/$dataset/enc$enc\_dec$dec/Lambda$epoch\_$lr/