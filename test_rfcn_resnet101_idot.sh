#!/usr/bin/env bash
#for SPLIT in train test;
#do
#  CUDA_VISIBLE_DEVICES=1 python3 forward_net.py \
#    --path_to_ckpt models/rfcn_resnet101_idot/frozen_inference_graph-91442.pb \
#    --labels data/idot_label_map.pbtxt \
#    --num_classes 2 \
#    --img_dir ~/darknet-finetune/IDOT_dataset/${SPLIT}/images \
#    --output_dir ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101_${SPLIT}/0.1 \
#    --threshold 0.1 \
#    --no-visualize \
#    --batch_size 8
#done
for SPLIT in train test;
do
  python3 ~/darknet-finetune/utils/convert_json_to_mot.py \
    ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101_${SPLIT}/0.1/json ~/IDOT_rfcn_resnet101_${SPLIT}@0.7.txt \
    --threshold 0.7
done
cat ~/IDOT_rfcn_resnet101_*@0.7.txt > ~/IDOT_rfcn_resnet101@0.7.txt
for SPLIT in train test;
do
  echo 'Precision Recall Curve for '${SPLIT}':'
  python3 ~/darknet-finetune/utils/compute_precision_recall.py \
    pascal_voc ~/darknet-finetune/IDOT_dataset/${SPLIT}/xml \
    json ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101_${SPLIT}/0.1/json \
    --output_path ~/IDOT_prec_rec_scores_rfcn_resnet101_${SPLIT}.pkl
  python3 ~/darknet-finetune/utils/create_precision_recall_curve.py ~/IDOT_prec_rec_scores_rfcn_resnet101_${SPLIT}.pkl
  echo '========================================================='
done
echo 'Precision Recall Curve for all '
python3 ~/darknet-finetune/utils/compute_precision_recall.py \
  pascal_voc ~/darknet-finetune/IDOT_dataset/xml \
  txt ~/IDOT_rfcn_resnet101@0.7.txt \
  --output_path ~/IDOT_prec_rec_scores_rfcn_resnet101.pkl
python3 ~/darknet-finetune/utils/create_precision_recall_curve.py ~/IDOT_prec_rec_scores_rfcn_resnet101.pkl
echo '========================================================='
