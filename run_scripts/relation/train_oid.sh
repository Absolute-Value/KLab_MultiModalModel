batch_size=64
for model in "google/flan-t5-small"
do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --image_model_train \
        --language_model_name google/flan-t5-base \
        --ffn \
        --transformer_model_name $model \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 20 \
        --save_interval 2 \
        --data_dir /data/dataset/openimage/ \
        --result_dir results/relation/
done