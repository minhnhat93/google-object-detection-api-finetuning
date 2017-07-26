CUDA_VISIBLE_DEVICES=1 python3 forward_net.py --path_to_ckpt models/rfcn_resnet101_idot/frozen_inference_graph_235663.pb --labels data/idot_label_map.pbtxt --num_classes 2 --img_dir ~/darknet-finetune/IDOT_dataset/images --output_dir ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101/0.1 --threshold 0.1 --no-visualize
#CUDA_VISIBLE_DEVICES=0 python3 forward_net.py --path_to_ckpt models/rfcn_resnet101_idot/frozen_inference_graph_164182.pb --labels data/idot_label_map.pbtxt --num_classes 2 --img_dir ~/darknet-finetune/IDOT_dataset/images --output_dir ~/out_test --threshold 0.1 --no-visualize
echo ' Precision Recall Curve'
python3 ~/darknet-finetune/utils/compute_precision_recall.py pascal_voc ~/darknet-finetune/IDOT_dataset/xml json ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101/0.1/json --output_path ~/IDOT_prec_rec_scores_rfcn_resnet101.pkl
python3 ~/darknet-finetune/utils/create_precision_recall_curve.py ~/IDOT_prec_rec_scores_rfcn_resnet101.pkl
echo '========================================================='
python3 ~/darknet-finetune/utils/convert_json_to_mot.py ~/darknet-finetune/IDOT_dataset/images/rfcn_resnet101/0.1/json ~/IDOT@0.1.txt

