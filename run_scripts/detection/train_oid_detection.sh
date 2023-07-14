batch_size=32
for model in "google/flan-t5-small"
do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --image_model_train \
        --language_model_name google/flan-t5-base \
        --ffn \
        --max_target_length 256 \
        --transformer_model_name $model \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 20 \
        --save_interval 5 \
        --data_dir /local_data1/openimage \
        --result_dir results/detection/fixdata/
done