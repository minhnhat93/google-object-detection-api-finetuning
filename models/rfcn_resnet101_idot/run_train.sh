#!/usr/bin/env bash
ROOT_DIR=/home/nhat/google-object-detection-api-finetuning/models/rfcn_resnet101_idot
CONFIG_FILE=${ROOT_DIR}/rfcn_resnet101_idot.config
CKPT_DIR=${ROOT_DIR}/train
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --logtostderr \
    --pipeline_config_path=${CONFIG_FILE} \
    --train_dir=${CKPT_DIR}

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_FILE} \
    --checkpoint_path ${CKPT_DIR}/model.ckpt \
    --inference_graph_path ${ROOT_DIR}/inference_graph.pb