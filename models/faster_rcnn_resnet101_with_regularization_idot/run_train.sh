#!/usr/bin/env bash
ROOT_DIR=/home/nhat/google-object-detection-api-finetuning/models/faster_rcnn_resnet101_with_regularization_idot
CONFIG_FILE=${ROOT_DIR}/faster_rcnn_resnet101_with_regularization_idot.config
CKPT_DIR=${ROOT_DIR}/train
EVAL_DIR=${ROOT_DIR}/eval
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --logtostderr \
    --pipeline_config_path=${CONFIG_FILE} \
    --train_dir=${CKPT_DIR}

CUDA_VISIBLE_DEVICES= python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_FILE} \
    --checkpoint_path ${CKPT_DIR}/model.ckpt-132860 \
    --inference_graph_path ${ROOT_DIR}/frozen_inference_graph.pb

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${CONFIG_FILE} \
    --checkpoint_dir=${CKPT_DIR} \
    --eval_dir=${EVAL_DIR}
