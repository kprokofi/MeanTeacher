#!/usr/bin/env bash
nvidia-smi

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export volna="/home/kprokofi/datasets/datasets"
# export NGPUS=4
# export OUTPUT_PATH="/local/kprokofi/segmentation/output_city_4_MT_16"
# export snapshot_dir=$OUTPUT_PATH/snapshot
# mkdir $OUTPUT_PATH

# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# export TARGET_DEVICE=$[$NGPUS-1]
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset city_4
# python eval.py -e 25-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset city_4 >> $OUTPUT_PATH/val.txt

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export volna="/home/kprokofi/datasets/datasets"
# export NGPUS=4
# export OUTPUT_PATH="/local/kprokofi/segmentation/output_city_MT_16"
# export snapshot_dir=$OUTPUT_PATH/snapshot

# mkdir $OUTPUT_PATH

# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# export TARGET_DEVICE=$[$NGPUS-1]
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset city
# python eval.py -e 25-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset city >> $OUTPUT_PATH/val.txt

# export CUDA_VISIBLE_DEVICES=0,1
# export volna="/home/kprokofi/datasets/datasets"
# export NGPUS=2
# export OUTPUT_PATH="/local/kprokofi/segmentation/output_disk_UPL_2"
# export snapshot_dir=$OUTPUT_PATH/snapshot

# mkdir $OUTPUT_PATH
# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# export TARGET_DEVICE=$[$NGPUS-1]
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset disk
# python eval.py -e 5-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset disk >> $OUTPUT_PATH/val.txt

export CUDA_VISIBLE_DEVICES=2,3
export volna="/home/kprokofi/datasets/datasets"
export NGPUS=2
export OUTPUT_PATH="/local/kprokofi/segmentation/output_voc_person_UPL_2"
export snapshot_dir=$OUTPUT_PATH/snapshot

mkdir $OUTPUT_PATH

export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1

export TARGET_DEVICE=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset voc_person
python eval.py -e 30-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset voc_person >> $OUTPUT_PATH/val.txt

# export volna="/home/kprokofi/datasets"
# export NGPUS=2
# export OUTPUT_PATH="/local/kprokofi/segmentation/output_kvasir_UPL_2"
# export snapshot_dir=$OUTPUT_PATH/snapshot

# mkdir $OUTPUT_PATH

# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset kvasir
# python eval.py -e 5-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset kvasir >> $OUTPUT_PATH/val.txt

# export volna="/home/kprokofi/datasets/datasets"
# export NGPUS=2
# export OUTPUT_PATH="/local/kprokofi/segmentation/output_fish_MT_16"
# export snapshot_dir=$OUTPUT_PATH/snapshot

# mkdir $OUTPUT_PATH

# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset fish
# python eval.py -e 25-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset fish >> $OUTPUT_PATH/val.txt

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # export volna="/home/kprokofi/datasets/datasets"
# # export NGPUS=4
# # export OUTPUT_PATH="/local/kprokofi/segmentation/output_voc_MT_16"
# # export snapshot_dir=$OUTPUT_PATH/snapshot

# # mkdir $OUTPUT_PATH

# # export batch_size=8
# # export learning_rate=0.0025
# # export snapshot_iter=1

# # export TARGET_DEVICE=$[$NGPUS-1]
# # python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --dev 0-$TARGET_DEVICE --dataset VOC
# # python eval.py -e 25-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset VOC >> $OUTPUT_PATH/val.txt
