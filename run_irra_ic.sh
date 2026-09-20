#!/bin/bash
DATASET_NAME="ICFG-PEDES"
DATASET_ROOT="/data/ljx/ydl/project/IRRA/IRRA-main/my_dataset_root"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--root_dir $DATASET_ROOT \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
