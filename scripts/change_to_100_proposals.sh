#!/usr/bin/env bash
pytpyhon3 /home/nhat/tensorflow-models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /home/nhat/tensorflow-models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/faster_rcnn_inception_resnet_v2_atrous_coco.config \
    --checkpoint_path /home/nhat/tensorflow-models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/model.ckpt \
    --inference_graph_path /home/nhat/tensorflow-models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/output_inference_graph_100_proposals.pb