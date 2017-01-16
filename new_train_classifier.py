#!/usr/bin/env python

import datetime
import json
import math
import numpy as np
import os
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 400,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("validate_batch_size", 2000,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "dnn",
                    "Model to train, option model: dnn, lr, wide_and_deep")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
flags.DEFINE_boolean("enable_dropout", True, "Enable dropout or not")
flags.DEFINE_float("dropout_keep_prob", 0.6, "The dropout keep prob")
flags.DEFINE_string("optimizer", "rmsprop", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 10,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Option mode: train, export, inference")
flags.DEFINE_string("output_path", "./output/", "indicates training output")
flags.DEFINE_string("model_path", "./model/", "indicates training output")
flags.DEFINE_integer("export_version", 1, "Version number of the model.")


def main():
  # Change these for different models
  FEATURE_SIZE = 440
  LABEL_SIZE = 8
  TRAIN_TFRECORDS_FILE = "data/train.tfrecords"
  VALIDATE_TFRECORDS_FILE = "data/test.tfrecords"

  learning_rate = FLAGS.learning_rate
  epoch_number = FLAGS.epoch_number
  thread_number = FLAGS.thread_number
  batch_size = FLAGS.batch_size
  validate_batch_size = FLAGS.validate_batch_size
  min_after_dequeue = FLAGS.min_after_dequeue
  capacity = thread_number * batch_size + min_after_dequeue
  mode = FLAGS.mode
  checkpoint_dir = FLAGS.checkpoint_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  tensorboard_dir = FLAGS.tensorboard_dir
  if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

  def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "x_train": tf.FixedLenFeature(
                                               [FEATURE_SIZE], tf.float32),
                                           "y_train": tf.FixedLenFeature(
                                               [LABEL_SIZE], tf.float32),
                                       })
    x_train = features["x_train"]
    y_train = features["y_train"]
    return x_train, y_train

  # Read TFRecords files for training
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(TRAIN_TFRECORDS_FILE),
      num_epochs=epoch_number)
  xtrain, ytrain = read_and_decode(filename_queue)
  batch_xtrain, batch_ytrain = tf.train.shuffle_batch(
      [xtrain, ytrain],
      batch_size=batch_size,
      num_threads=thread_number,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  # Read TFRecords file for validatioin
  validate_filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(VALIDATE_TFRECORDS_FILE),
      num_epochs=epoch_number)
  xtest, ytest = read_and_decode(validate_filename_queue)
  batch_xtest, batch_ytest = tf.train.shuffle_batch(
      [xtest, ytest],
      batch_size=validate_batch_size,
      num_threads=thread_number,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)


  # Define the model
  input_units = FEATURE_SIZE
  hidden1_units = 500
  hidden2_units = 500
  hidden3_units = 500
  output_units = LABEL_SIZE

  def full_connect(inputs, weights_shape, biases_shape, is_train=True):
    with tf.device('/cpu:0'):
      weights = tf.get_variable("weights",
                                weights_shape,
                                initializer=tf.random_normal_initializer())
      biases = tf.get_variable("biases",
                               biases_shape,
                               initializer=tf.random_normal_initializer())
      layer = tf.matmul(inputs, weights) + biases

      if FLAGS.enable_bn and is_train:
        mean, var = tf.nn.moments(layer, axes=[0])
        scale = tf.get_variable("scale",
                                biases_shape,
                                initializer=tf.random_normal_initializer())
        shift = tf.get_variable("shift",
                                biases_shape,
                                initializer=tf.random_normal_initializer())
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                          FLAGS.bn_epsilon)
    return layer

  def full_connect_relu(inputs, weights_shape, biases_shape, is_train=True):
    layer = full_connect(inputs, weights_shape, biases_shape, is_train)
    layer = tf.nn.relu(layer)
    return layer

  def dnn_inference(inputs, is_train=True):

    with tf.variable_scope("layer1"):
      layer = full_connect_relu(inputs, [input_units, hidden1_units],
                                [hidden1_units], is_train)
    if FLAGS.enable_dropout and is_train:
      layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)

    with tf.variable_scope("layer2"):
      layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                                [hidden2_units], is_train)
    if FLAGS.enable_dropout and is_train:
      layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)

    with tf.variable_scope("layer3"):
      layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                                [hidden3_units], is_train)
    if FLAGS.enable_dropout and is_train:
      layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)

    with tf.variable_scope("output"):
      layer = full_connect(layer, [hidden3_units, output_units],
                           [output_units], is_train)

      layer = tf.nn.sigmoid(layer)
    return layer

  def lr_inference(inputs, is_train=True):
    with tf.variable_scope("logistic_regression"):
      layer = full_connect(inputs, [input_units, output_units], [output_units])
    return layer

  def wide_and_deep_inference(inputs, is_train=True):
    return lr_inference(inputs, is_train) + dnn_inference(inputs, is_train)

  def inference(inputs, is_train=True):
    print("Use the model: {}".format(FLAGS.model))
    if FLAGS.model == "lr":
      return lr_inference(inputs, is_train)
    elif FLAGS.model == "dnn":
      return dnn_inference(inputs, is_train)
    elif FLAGS.model == "wide_and_deep":
      return wide_and_deep_inference(inputs, is_train)
    else:
      print("Unknown model, exit now")
      exit(1)

  logits = inference(batch_xtrain, True)
#  batch_ytrain = tf.to_int64(batch_ytrain)
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, batch_ytrain)
  loss = tf.reduce_mean(cross_entropy, name='loss')

  print("Use the optimizer: {}".format(FLAGS.optimizer))
  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
    exit(1)

  with tf.device("/cpu:0"):
    global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  tf.get_variable_scope().reuse_variables()


  # Define accuracy op for train data, with all train data or batch size data
  train_accuracy_logits = inference(batch_xtrain, False)
#  train_sigmoid = tf.nn.sigmoid(train_accuracy_logits)
  train_sigmoid = train_accuracy_logits
  train_correct_prediction = tf.equal(
      tf.argmax(train_sigmoid, 1), tf.argmax(batch_ytrain, 1))
  train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

  # Define accuracy op for validate data
  validate_accuracy_logits = inference(batch_xtest, False)
#  validate_sigmoid = tf.nn.sigmoid(validate_accuracy_logits)
  validate_sigmoid = validate_accuracy_logits
  # batch_ytest = tf.to_int64(batch_ytest)
  validate_correct_prediction = tf.equal(
      tf.argmax(validate_sigmoid, 1), tf.argmax(batch_ytest, 1))
  validate_accuracy = tf.reduce_mean(tf.cast(validate_correct_prediction, tf.float32))

  # Define inference op
  inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
  inference_logits = inference(inference_features, False)
#  inference_sigmoid = tf.nn.sigmoid(inference_logits)
  inference_sigmoid = inference_logits
  inference_op = tf.argmax(inference_sigmoid, 1)


  # Initialize saver and summary
  checkpoint_file = checkpoint_dir + "checkpoint.ckpt"
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  steps_to_validate = FLAGS.steps_to_validate
  init_op = tf.initialize_all_variables()
  tf.scalar_summary("loss", loss)
  tf.scalar_summary("train_accuracy", train_accuracy)
#  tf.scalar_summary("train_auc", train_auc)
  tf.scalar_summary("validate_accuracy", validate_accuracy)
#  tf.scalar_summary("validate_auc", validate_auc)
  saver = tf.train.Saver()
  keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
  keys = tf.identity(keys_placeholder)
  tf.add_to_collection("inputs",
                       json.dumps({'key': keys_placeholder.name,
                                   'features': inference_features.name}))
  tf.add_to_collection("outputs",
                       json.dumps({'key': keys.name,
                                   'sigmoid': inference_sigmoid.name,
                                   'prediction': inference_op.name}))

  # Create session to run
  with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
    sess.run(init_op)
    sess.run(tf.initialize_local_variables())

    if mode == "train":
      if latest_checkpoint:
        print("Load the checkpoint from {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

      # Get coordinator and run queues to read data
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)

      start_time = datetime.datetime.now()
      try:
        while not coord.should_stop():
          _, loss_value, step = sess.run([train_op, loss, global_step])

          if step % steps_to_validate == 0:
#            train_accuracy_value, train_auc_value, validate_accuracy_value, validate_auc_value, summary_value = sess.run(
#                [train_accuracy, train_auc, validate_accuracy, validate_auc,
#                 summary_op])
            train_accuracy_value, validate_accuracy_value,  summary_value = sess.run(
                [train_accuracy, validate_accuracy, summary_op])

            end_time = datetime.datetime.now()
            print(
                "[{}] Step: {}, loss: {}, train_acc: {},  valid_acc: {}".format(
                    end_time - start_time, step, loss_value,
                    train_accuracy_value, validate_accuracy_value))

            writer.add_summary(summary_value, step)
            saver.save(sess, checkpoint_file, global_step=step)
            start_time = end_time
      except tf.errors.OutOfRangeError:
        print("Done training after reading all data")
        print("Exporting trained model to {}".format(FLAGS.model_path))
        model_exporter = exporter.Exporter(saver)
        model_exporter.init(sess.graph.as_graph_def(),
                            named_graph_signatures={
                                'inputs': exporter.generic_signature(
                                    {"keys": keys_placeholder,
                                     "features": inference_features}),
                                'outputs': exporter.generic_signature(
                                    {"keys": keys,
                                     "sigmoid": inference_sigmoid,
                                     "prediction": inference_op})
                            })
        model_exporter.export(FLAGS.model_path,
                              tf.constant(FLAGS.export_version), sess)
        print 'Done exporting!'

      finally:
        coord.request_stop()
      # Wait for threads to exit
      coord.join(threads)


    elif mode == "export":
      print("Start to export model directly")
      # Load the checkpoint files
      if latest_checkpoint:
        print("Load the checkpoint from {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
      else:
        print("No checkpoint found, exit now")
        exit(1)

      # Export the model files
      print("Exporting trained model to {}".format(FLAGS.model_path))
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(sess.graph.as_graph_def(),
                          named_graph_signatures={
                              'inputs': exporter.generic_signature(
                                  {"keys": keys_placeholder,
                                   "features": inference_features}),
                              'outputs': exporter.generic_signature(
                                  {"keys": keys,
                                   "sigmoid": inference_sigmoid,
                                   "prediction": inference_op})
                          })
      model_exporter.export(FLAGS.model_path,
                            tf.constant(FLAGS.export_version), sess)

    elif mode == "inference":
      print("Start to run inference")
      start_time = datetime.datetime.now()

      inference_result_file_name = "./inference_result.txt"
      inference_test_file_name = "./data/cancer_test.csv"

      inference_data = np.genfromtxt(inference_test_file_name, delimiter=',')
      inference_data_features = inference_data[:, 0:9]
      inference_data_labels = inference_data[:, 9]

      # Restore wights from model file
      if latest_checkpoint:
        print("Load the checkpoint from {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
      else:
        print("No model found, exit now")
        exit(1)

      prediction, prediction_sigmoid = sess.run(
          [inference_op, inference_sigmoid],
          feed_dict={inference_features: inference_data_features})

      end_time = datetime.datetime.now()
      print("[{}] Inference result: {}".format(end_time - start_time,
                                               prediction))

      # Compute accuracy
      label_number = len(inference_data_labels)
      correct_label_number = 0
      for i in range(label_number):
        if inference_data_labels[i] == prediction[i]:
          correct_label_number += 1
      accuracy = float(correct_label_number) / label_number

      # Compute auc
      expected_labels = np.array(inference_data_labels)
      predict_labels = prediction_sigmoid[:, 0]
      fpr, tpr, thresholds = metrics.roc_curve(expected_labels,
                                               predict_labels,
                                               pos_label=0)
      auc = metrics.auc(fpr, tpr)
      print("For inference data, accuracy: {}, auc: {}".format(accuracy, auc))

      np.savetxt(inference_result_file_name, prediction, delimiter=",")
      print("Save result to file: {}".format(inference_result_file_name))


if __name__ == "__main__":
  main()