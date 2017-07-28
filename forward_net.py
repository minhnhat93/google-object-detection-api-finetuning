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

COLOR_MAPS = [
  (255, 62, 150),
  (0, 250, 154),
  (128, 128, 105),
  (255, 128, 0),
  (139, 26, 26),
  (198, 113, 113),
  (0, 206, 209),
  (178, 58, 238),
  (255, 20, 147),
  (238, 169, 184),
  (102, 205, 170),
  (179, 238, 58),
  (205, 205, 0),
]


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
  parser = argparse.ArgumentParser(description='script used to forward model at tensorflow/models/object-detection in'
                                               ' batches')
  parser.add_argument('--path_to_ckpt', help='path to frozen graph', type=str)
  parser.add_argument('--labels', help='path to label pbtxt', type=str)
  parser.add_argument('--num_classes', help='number of classes in pbtxt file', type=int)
  parser.add_argument('--img_dir', help='input directory contains images to forward', type=str)
  parser.add_argument('--output_dir', default=None, type=str)
  parser.add_argument('--file_ext', default='jpg', type=str)
  parser.add_argument('--threshold', default=0.1, help='objectness score threshold', type=float)
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
      # get batch size, height, width of image
      batch_size = image_tensor._shape_as_list()[0]
      dummy_imgs = np.random.random((batch_size, 1, 1, 3))
      reshaped_img_tensor = detection_graph.get_tensor_by_name(
        'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0')
      _, img_height, img_width, _ = sess.run(reshaped_img_tensor, feed_dict={image_tensor: dummy_imgs}).shape
      # index of image being processed in current iteration
      image_index = 0
      while True:
        # init values for current batch
        if image_index % batch_size == 0:
          start = timeit.default_timer()
          curr_batch = []  # resized images
          raw_img_in_batch = []  # original images
          img_path_in_batch = []  # path of images
          num_img_in_batch = 0
        # put image into image_tensor
        if image_index < len(images_set):
          image_path = images_set[image_index]
          image_np = cv2.imread(os.path.join(args.img_dir, image_path))[:, :, ::-1]
          raw_img_in_batch.append(image_np)
          img_path_in_batch.append(image_path)
          # resize image to a fixed size to process in batches
          image_np = cv2.resize(image_np, (img_height, img_width))
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          num_img_in_batch += 1
        else:
          # if no more image fill tensor with 0 values
          image_np_expanded = np.zeros((1, img_height, img_width, 3))
        curr_batch.append(image_np_expanded)
        image_index += 1

        # every batch_size images, we do inference 
        if len(curr_batch) == batch_size:
          # Actual detection.
          curr_batch = np.vstack(curr_batch)
          start_detection = timeit.default_timer()
          (batch_boxes, batch_scores, batch_classes, batch_num_detections) = sess.run(
            [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
            feed_dict={image_tensor: curr_batch})
          stop = timeit.default_timer()
          print('Inference {}: {:.6f}s / {} inputs = {:.2f} fps'.format(image_path, stop - start_detection, batch_size,
                                                                        float(batch_size) / (stop - start_detection)))
          # Visualization of the results of a detection.
          for j in range(num_img_in_batch):
            boxes = np.squeeze(batch_boxes[j])
            classes = np.squeeze(batch_classes[j]).astype(np.int32)
            scores = np.squeeze(batch_scores[j])
            image_np = raw_img_in_batch[j][:, :, ::-1]
            image_path = img_path_in_batch[j]
            height, width, _ = image_np.shape
            thick = int((height + width) // 300)
            results_for_json = []
            for box, cls, score in zip(boxes, classes, scores):
              if score < args.threshold:
                break
              class_name = index_to_name_map[cls]
              ymin = int(box[0] * height)
              xmin = int(box[1] * width)
              ymax = int(box[2] * height)
              xmax = int(box[3] * width)
              if args.save_json:
                results_for_json.append({"label": class_name, "confidence": float('%.2f' % score),
                                         "topleft": {"x": xmin, "y": ymin}, "bottomright": {"x": xmax, "y": ymax}})
              if args.visualize:
                color = COLOR_MAPS[cls % len(COLOR_MAPS)]
                cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, thick)
                cv2.putText(image_np, class_name + ' {}%'.format(int(score * 100)), (xmin, ymin - 10), 0, 1e-3 * height,
                            color, thick // 2)
            if args.visualize:
              cv2.imwrite(os.path.join(VISUALIZATION_DIR, os.path.basename(image_path)), image_np)
            if args.save_json:
              json.dump(results_for_json, open(os.path.join(JSON_DIR, os.path.splitext(image_path)[0] + '.json'), 'w'))
          stop = timeit.default_timer()
          print('Total: {:6f} / {} inputs = {:.2f} fps'.format(stop - start, num_img_in_batch,
                                                               float(num_img_in_batch) / (stop - start)))
          if image_index >= len(images_set):
            break
