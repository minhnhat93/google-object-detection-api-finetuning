import tensorflow as tf

from utils import dataset_util, label_map_util
from utils.annotation_parsing import parse_pascal_voc_groundtruth
import os
import numpy as np


flags = tf.app.flags
flags.DEFINE_string('image_dir', '/home/nhat/darknet-finetune/IDOT_dataset/images', 'Path to images dir')
flags.DEFINE_string('annotation_dir', '/home/nhat/darknet-finetune/IDOT_dataset/xml', 'Path to xml dir')
flags.DEFINE_string('imageset_txt', '/home/nhat/darknet-finetune/idot_finetune/idot_train_index.txt',
                    'text file contains train/test index. if applicable')
flags.DEFINE_string('label_map_path', 'data/idot_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('output_path', './data/idot_train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# Create test record:
# python3 convert_datasets/convert_idot_to_tfrecord.py --imageset_txt /home/nhat/darknet-finetune/idot_finetune/idot_train_index.txt --output_path ./data/idot_test.record

def create_tf_example(frame, label_map_dict):
  # TODO(user): Populate the following variables from your example.
  height = frame['height'] # Image height
  width = frame['width'] # Image width
  filename = '{}.jpg'.format(frame['frame_id']) # Filename of the image. Empty if image is not from file
  img_path = os.path.join(FLAGS.image_dir, filename)
  filename = filename.encode()
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_image_data = fid.read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [float(bbox[0]) / width for bbox in frame['bboxes']] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [float(bbox[2]) / width for bbox in frame['bboxes']] # List of normalized right x coordinates in bounding box
  # (1 per box)
  ymins = [float(bbox[1]) / height for bbox in frame['bboxes']] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [float(bbox[3]) / height for bbox in frame['bboxes']] # List of normalized bottom y coordinates in bounding box
  # (1 per box)
  classes_text = [name.encode() for name in frame['names']] # List of string class name of bounding box (1 per box)
  classes = [label_map_dict[name] for name in frame['names']] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable

  frames = parse_pascal_voc_groundtruth(FLAGS.annotation_dir)
  if FLAGS.imageset_txt != '':
    train_set = np.genfromtxt(FLAGS.imageset_txt, dtype=np.int)
  else:
    train_set = None
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  for frame in frames.values():
    if train_set is None or int(frame['frame_id']) in train_set:
      tf_example = create_tf_example(frame, label_map_dict)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
