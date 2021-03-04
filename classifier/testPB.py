

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import cv2

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=180, input_width=180,
				input_mean=0, input_std=1):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  model_file = "/media/thanglmb/Bkav/AICAM/TrainModels/TrainedModel/Hat/model_hat.pb"
  label_file = "/media/thanglmb/Bkav/AICAM/TrainModels/TrainedModel/Hat/labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"
  cam = "cam"
  parser = argparse.ArgumentParser()
  parser.add_argument("--cam", help="open camera with cmd: \"cam\" ")
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.cam:
    cap = cv2.VideoCapture(0)
  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer



  graph = load_graph(model_file)
  while(True):
    
    # print(t)
    # print("---------------------------------------------------------")
    # pre-processing
    if args.cam:
      ret, frame = cap.read()
      img_resize = cv2.resize(frame , (input_width,input_height))
      #frame = cv2.imread("/media/thanglmb/Bkav/AICAM/SNPE/Issues/ReducedAccuracy/datatest/1.png", 1)
      img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
      
      # img_tensor = np.asarray(img_resize)
      # img_tensor = np.expand_dims(img_resize, axis=0)
      # print(img_tensor)
      image_tensor = tf.convert_to_tensor(img_resize, dtype=tf.uint8)
      float_caster = tf.cast(image_tensor, tf.float32)
      dims_expander = tf.expand_dims(float_caster, axis=0)
      resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
      img_tensor = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      sess = tf.Session()
      result = sess.run(img_tensor)
      img_tensor = result
    else:
      img_tensor = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)
    
    #print(img_tensor)
    # excute network
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: img_tensor})
      end=time.time()
    print("result:", results)
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    print("----------------------------------------------------")
    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    # get result
    if args.image:
      frame = cv2.imread(file_name, 1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame , (640,480))
    for i in top_k:
      print(template.format(labels[i], results[i]))
      
    # Display the resulting frame 
    cv2.putText(frame, template.format(labels[np.argmax(results)], 100 * np.max(results)), (10,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break 
