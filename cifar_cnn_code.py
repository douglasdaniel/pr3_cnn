#!/usr/bin/env python3

# Author: Daniel Douglas


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle, pdb, cv2

def import_cifar(type):
    # Import data
    with open("data/cifar_10_tf_train_test.pkl","rb") as data_file:
        train_x,train_y,test_x,test_y = pickle.load(data_file,encoding='latin1')
    train_x = train_x.astype(np.float32) / 255 
    test_x = test_x.astype(np.float32) / 255
    train_y = np.asarray(train_y,dtype=np.int32)
    test_y = np.asarray(test_y,dtype=np.int32)
    
    # Return dataset
    if type == 'train':
        return train_x,train_y,
    elif type == 'test':
        return test_x,test_y
    else:
        return train_x,train_y,test_x,test_y

# Enable informative logging
tf.logging.set_verbosity(tf.logging.INFO)

# Define model architecture
def cnn_model(features, labels, mode):
    input_layer = features["x"]

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="valid",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    # Convolutional Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="valid",
        activation = tf.nn.relu)

    # Full-Connected Layer
    flat1 = tf.reshape(conv3, [-1, conv3.shape[1]*conv3.shape[2]*64])

    # Logits Layer
    logits = tf.layers.dense(inputs=flat1, units=10)

    predictions = {
        "classes":tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    accurate = tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Main function
def main(unused_argv):

    # Load training and testing data
    train_data, train_labels = import_cifar('train')
    test_data, test_labels = import_cifar('test')

    # Create estimator
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="./outputs/cifar_cnn_model")

    # Set up logging
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Model training
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Saving model
    x = tf.placeholder(shape=[None,32,32,3], dtype=tf.float32)
    y = tf.placeholder(shape=[None], dtype=tf.int32)
    predict_op = "cnn_model(features,labels,PREDICT)"

    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", x)
    tf.add_to_collection("validation_nodes", y)
    tf.add_to_collection("validation_nodes", predict_op)

    cifar_classifier.train(
        input_fn=train_input,
        steps=20000,
        hooks=[logging_hook])
        
    # Evaluate model performance
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y= test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results= cifar_classifier.evaluate(input_fn=eval_input)
    print(eval_results)
        
if __name__ == "__main__":
    tf.app.run()

    
    

