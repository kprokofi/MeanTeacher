nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export batch_size=8
export learning_rate=0.003
export snapshot_iter=1
export TARGET_DEVICE=$[$NGPUS-1]
###

export volna="/home/kprokofi/datasets/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_city_4_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset city_4
python eval.py -e 10-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset city_4 >> $OUTPUT_PATH/val.txt

export volna="/home/kprokofi/datasets/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_city_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset city
python eval.py -e 10-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset city >> $OUTPUT_PATH/val.txt

export volna="/home/kprokofi/datasets/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_disk_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset disk
python eval.py -e 5-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset disk >> $OUTPUT_PATH/val.txt

export volna="/home/kprokofi/datasets/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_voc_person_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset voc_person
python eval.py -e 5-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset voc_person >> $OUTPUT_PATH/val.txt

export volna="/home/kprokofi/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_kvasir_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH
export learning_rate=0.0035

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset kvasir
python eval.py -e 5-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset kvasir >> $OUTPUT_PATH/val.txt

export volna="/home/kprokofi/datasets/datasets"
export OUTPUT_PATH="/local/kprokofi/segmentation/output_fish_AEL_finilized_1to16"
export snapshot_dir=$OUTPUT_PATH/snapshot
mkdir $OUTPUT_PATH

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 10105 train.py --dev 0-$TARGET_DEVICE --dataset fish
python eval.py -e 25-40 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results --dataset fish >> $OUTPUT_PATH/val.txt
