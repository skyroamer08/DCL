gpu=0
lr=1e-6
experiments=ex0
dataset=f30k    # f30k/coco
dataset_root=''
noise_ratio=0.4
batch_size=128
epochs=5
balance1=0.2
balance2=128
dev_length=5000
num_anns=5
threshold=0.5


CUDA_VISIBLE_DEVICES=$gpu python main.py --lr $lr --experiments $experiments \
    --dataset $dataset --dataset_root $dataset_root --noise_ratio $noise_ratio \
    --batch_size $batch_size --epochs $epochs --balance1 $balance1 --balance2 $balance2 \
    --dev_length $dev_length --num_anns $num_anns --threshold $threshold
