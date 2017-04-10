#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: vickylee
"""
# CSC 578 Project 3 Vicky Lee
# Convolutional Neural Network on MNIST with Modifications for Tensorboard Summary Data Visualization 

# Import the modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import tensorflow module
import tensorflow as tf

# download and read in MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Create new directory with path '/logs' after deletign existing files if present within '/logs' directory
if tf.gfile.Exists('/logs'):
    tf.gfile.DeleteRecursively('/logs')
tf.gfile.MakeDirs('/logs')

# Sets a new default graph and stores it in g
with tf.Graph().as_default() as g: 
    # Assign mnist as a lightweight class which stores the training, validation, and testing sets
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # Create tensorflow interactive session
    sess = tf.InteractiveSession()
    
    # Create name scope for the the follwing op as 'input'
    with tf.name_scope('input'):
        # Create nodes for the input images and target output classes as 2d tensor through Placeholders
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        
    # Create name scope for the follwing op as 'input_reshape'
    with tf.name_scope('input_reshape'):
        # Before applying the layer, reshape x to a 4d tensor with 2nd and 3rd dimensions of image width and 
        # height and 4th dimension as the number of color channels
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', image_shaped_input, 10)
        
    # Define functions weights and biases that take in the parameter shape which determines matrix size of 
    # weights and biases
    # Initialize weights as random truncated normal values with 0 mean and 0.1 standard deviation 
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # Initialize biases as contant value of 0.1
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    # Define function of convolutions as using a stride of one and zero padded so the output is the same size as input 
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # Define function of pooling as max pooling over 2x2 blocks
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    # Define function variable_summaries that takes in parmaeter var and compute follwing statistics and 
    # produce reprot with name tag
    def variable_summaries(var, name):
      # Create name scope for the follwing op as 'summaries'
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # Generate mean of var tagged as 'mean/' + name
        tf.scalar_summary('mean/' + name, mean)
        # Create name scope for the follwing op as 'stddev'
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # Generate stddev tagged as 'stddev/' + name
        tf.scalar_summary('stddev/' + name, stddev)
        # Generate maximum of var tagged as 'max/' + name
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        # Generate minimum of var tagged as 'min/' + name
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        # Generate histogram of var tagged as name
        tf.histogram_summary(name, var)
    
    # Define convolution that takes in 6 parameter of input_tensor, input_dim, output_dim, layer_name, 
    # whether to apply convolution to or multiply x with weights, and type of activation function
    def conv(input_tensor, input_dim, output_dim, layer_name, apply=conv2d, act=tf.nn.relu):
      # Create name scope for the following op as 'layer_name'
      with tf.name_scope(layer_name):
        # Create name scope for the follwing op as 'wegihts'
        with tf.name_scope('weights'):
          # Initialize weights for the given dimension
          weights = weight_variable(input_dim + output_dim)
          # Generate variable_summaries for weights and tag it as layer_name + '/weights' 
          variable_summaries(weights, layer_name + '/weights')
        # Create name scope for the follwing op as 'biases'
        with tf.name_scope('biases'):
          # Initialize biases for the given dimension
          biases = bias_variable(output_dim)
          # Generate variable_summaries for biases and tag it as layer_name + '/biases'
          variable_summaries(biases, layer_name + '/biases')
        # Create name scope for the follwing op as 'Wx_plus_b'
        with tf.name_scope('Wx_plus_b'):
          # Apply either conv2d(convolutional layer) to or tf.matmul(multiply) input_tensor 
          # with the weight tensor, add the bias
          preactivate = apply(input_tensor, weights) + biases
          # Generate histogram of pre-activations tagged as layer_name + '/pre_activations'
          tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        # Apply ReLU function
        activations = act(preactivate, name='activation')
        # Generate histogram of activations tagged as layer_name + '/activations'
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations

    # Run first layer convolution as computing 32 features for each 5x5 patch 
    # and apply relu activation function
    convlayer1 = conv(image_shaped_input, [5, 5, 1], [32], 'conv1', apply=conv2d,  act=tf.nn.relu)

    # Create name scope for the follwing op as 'maxpool1'
    with tf.name_scope('maxpool1'):
        # Apply maxpool over 2x2 blocks
        maxpool1 = max_pool_2x2(convlayer1)
        # Generate histogram of maxpool1 tagged as maxpool1
        tf.histogram_summary('maxpool1',maxpool1)

    # Run second layer convolution as computing 64 features for each 5x5 patch 
    # and apply relu activation function
    convlayer2 = conv(maxpool1, [5, 5, 32], [64], 'conv2', apply=conv2d, act=tf.nn.relu)
    
    # Create name scope for the follwing op as 'maxpool2'
    with tf.name_scope('maxpool2'):
        # Apply maxpool over 2x2 blocks
        maxpool2 = max_pool_2x2(convlayer2)
        # Output histogram of maxpool2 tagged as maxpool2
        tf.histogram_summary('maxpool2',maxpool2)

    # Create name scope for the following op as 'input_flat'
    with tf.name_scope('input_flat'):
        # Reshape the tensor from the pooling layer into a batch of vectors
        flat = tf.reshape(maxpool2, [-1, 7*7*64])

    # Add a fully-connected layer with 1024 neurons to process the entire image by multiplying
    # reshpaed tensor by a weight matrix, adding a bias, and applying ReLU
    fulllayer = conv(flat,[7 * 7 * 64],[1024], 'full', apply=tf.matmul, act=tf.nn.relu)
    
    # Create name scope for the follwing op as 'dropout'
    with tf.name_scope('dropout'):
      # Create a placeholder for the probability that a neuron's output is kept during dropout 
      # which allows us to turn dropout on during training, and turn it off during testing 
      keep_prob = tf.placeholder(tf.float32)
      # Generate probability that a neuron's output is kept during dropout
      tf.summary.scalar('dropout_keep_probability', keep_prob)
      # Apply dropout before the readout layer by tf.nn.dropout op which automatically 
      # handles scaling neuron outputs in addition to masking them  
      dropped = tf.nn.dropout(fulllayer, keep_prob)

    # Add a readout layer by multiplying post-dropout tensor by a weight matirx, add a bias
    y_conv = conv(dropped, [1024], [10], 'output', apply=tf.matmul, act=tf.identity)

    # Create name scope for the follwing op as 'cross_entropy'
    with tf.name_scope('cross_entropy'):
        # Define loss function as cross-entropy between the target and the softmax activation 
        # function applied to the model's prediction
      diff = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
        # Create name scope for the follwoing op as 'total'
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    # Generate cross_entropy tagged as 'cross_entropy'
    tf.scalar_summary('cross_entropy', cross_entropy)

    # Create name scope for the following op as 'train'
    with tf.name_scope('train'):
        # Set train_step as using the ADAM optimizer with step length of 1e-4
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Create name scope for the follwoing op as 'accuracy'
    with tf.name_scope('accuracy'): 
      # Create name scope for the follwoing op as 'correct_prediction'  
      with tf.name_scope('correct_prediction'):
        # Identify correct prediction by comparing model output and target label
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        # Compute the percentage of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Generate accuracy tagged as 'accuracy'
    tf.scalar_summary('accuracy', accuracy)
    
    # Combine all summary ops into a single op, merged that generates all the summary data
    merged = tf.merge_all_summaries()
    # Write summary data to '/logs/train'
    train_writer = tf.train.SummaryWriter('/logs'+ '/train',sess.graph)
    # Write summary data to '/logs/test'
    test_writer = tf.train.SummaryWriter('/logs'+ '/test')
    # Initialize Variables Weights & biases for the session
    tf.initialize_all_variables().run()

    # Define feed_dict to replace tensor placeholders x, y_ with the follwoing 
    def feed_dict(train):
      # For train data, replace with new batch of training examples with dropout probability of 0.5
      if train:
        xs, ys = mnist.train.next_batch(100)
        k = 0.5
      # For test, repalce with all test samples
      else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
      return {x: xs, y_: ys, keep_prob: k}

    # Run for 20,000 epochs
    for i in range(20000):
      # For every 10th step
      if i % 10 == 0:  
        # Run merged and accuracy ops for test set to record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        # Write test summary every 10th stpes
        test_writer.add_summary(summary, i)
        # Print test accuracy every 10th stpes
        print('Accuracy at step %s: %s' % (i, acc))
      # For all other steps
      else:  
        # Run merged and train_stpes ops for train set to record train set summarieis and train
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        # Write train summary 
        train_writer.add_summary(summary, i)
    
    # Close train and test_writers
    train_writer.close()
    test_writer.close()


