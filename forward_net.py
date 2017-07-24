import numpy as np
import timeit
import os
import tensorflow as tf
import json
import glob

from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util

import argparse
import cv2


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


def get_all_images_in_dir(path, file_pattern='*.jpg'):
  curr_dir = os.getcwd()
  os.chdir(path)
  files = os.listdir('.')
  files = glob.glob(str(files) + file_pattern)
  os.chdir(curr_dir)
  return files


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_ckpt', help='path to frozen graph', type=str)
  parser.add_argument('--labels', help='path to label pbtxt', type=str)
  parser.add_argument('--num_classes', type=int)
  parser.add_argument('--img_dir', type=str)
  parser.add_argument('--output_dir', default=None, type=str)
  parser.add_argument('--file_ext', default='jpg', type=str)
  parser.add_argument('--threshold', default=0.1, type=float)
  parser.add_argument('--visualize', dest='visualize', action='store_true')
  parser.add_argument('--no-visualize', dest='visualize', action='store_false')
  parser.set_defaults(visualize=True)
  parser.add_argument('--save_json', dest='save_json', action='store_true')
  parser.add_argument('--no-save_json', dest='save_json', action='store_false')
  parser.set_defaults(save_json=True)
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  if args.output_dir is None:
    args.output_dir = args.img_dir

  VISUALIZATION_DIR = os.path.join(args.output_dir, 'visualization')
  JSON_DIR = os.path.join(args.output_dir, 'json')

  if args.visualize and not os.path.exists(VISUALIZATION_DIR):
    os.system('mkdir -p {}'.format(VISUALIZATION_DIR))
  if args.save_json and not os.path.exists(JSON_DIR):
    os.system('mkdir -p {}'.format(JSON_DIR))

  label_map = label_map_util.load_labelmap(args.labels)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args.num_classes,
                                                              use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  index_to_name_map = dict((category['id'], category['name']) for category in categories)
  detection_graph = tf.Graph()

  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.path_to_ckpt, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  images_set = get_all_images_in_dir(args.img_dir, file_pattern='*.{}'.format(args.file_ext))
  # Size, in inches, of the output images.

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
      classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')
      for image_path in images_set:
        start_total = timeit.default_timer()
        image_np = cv2.imread(os.path.join(args.img_dir, image_path))[:, :, ::-1]
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        start = timeit.default_timer()
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
          feed_dict={image_tensor: image_np_expanded})
        stop = timeit.default_timer()
        print('Inference {}: {:.6f}s ~ {:.2f} fps'.format(image_path, stop - start, 1.0 / (stop - start)))
        # Visualization of the results of a detection.
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        results_for_json = []
        height, width, _ = image_np.shape
        for box, cls, score in zip(boxes, classes, scores):
          if score < args.threshold:
            break
          class_name = index_to_name_map[cls]
          ymin = int(box[0] * height)
          xmin = int(box[1] * width)
          ymax = int(box[2] * height)
          xmax = int(box[3] * width)
          if args.save_json:
            results_for_json.append({"label": class_name, "confidence": float('%.2f' % score), "topleft": {"x": xmin, "y": ymin}, "bottomright": {"x": xmax, "y": ymax}})
        if args.visualize:
          vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=args.threshold)
          out_img = Image.fromarray(image_np)
          out_img.save(os.path.join(VISUALIZATION_DIR, os.path.basename(image_path)))
        if args.save_json:
          json.dump(results_for_json, open(os.path.join(JSON_DIR, os.path.splitext(image_path)[0] + '.json'), 'w'))
        stop = timeit.default_timer()
        print('Total: {:6f} ~ {:.2f} fps'.format(stop - start_total, 1.0 / (stop - start_total)))
