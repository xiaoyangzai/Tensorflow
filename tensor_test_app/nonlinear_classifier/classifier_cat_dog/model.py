#!/usr/bon/python
#-*- coding: utf-8 -*-

import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod

class BaseModel(object):
    """ Abstract base class for any model. """
    __metaclass__ = ABCMeta

    def __init__(self, num_classes):
        self._num_classes = num_classes

    @property
    def num_classes(self):
        return self._num_classes
    
    @abstractmethod
    def preprocess(self,inputs):
        """ Input preprocessing. To be override by implementations.

        Args:
            inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.

        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.
        
        """
        pass
    @abstractmethod
    def predict(self,preprocessed_inputs):
        """
            Predict prediction tensors from inputs tensor.
            Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be passed to the loss or postprocess functions.
        """
        pass

    @abstractmethod
    def postprocess(self,prediction_dict,**params):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors to be passed to the loss or postprocess functions.
            groundtruth_lists: A list of tensors holding groundtruth information, with one entry for each image in the batch.
        
        Returns:
            A dictionary containing the postprocessed results.
        """
        pass

    @abstractmethod
    def loss(self, prediction_dict,groundtruth_lists):
        """Compute the scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors representing loss values.
        """
        pass

class Model(BaseModel):
    """A simple 10-classification CNN model definition. """

    def __init__(self,is_training,num_classes):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(Model, self).__init__(num_classes = num_classes)
        self._is_training = is_training

    def preprocess(self,inputs):
        """ Input preprocessing. To be override by implementations.

        Args:
            inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.

        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.
        """
        preprocessed_inputs = tf.to_float(inputs)
        vgg_mean = tf.constant([123.68,116,779,103.939],dtype=tf.float32,shape=(1,1,1,3),name='vgg_mean')
        preprocessed_inputs -= vgg_mean 
        return preprocessed_inputs

    def predict(self, preprocessed_inputs):
        """
            Predict prediction tensors from inputs tensor.
            Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,height,width,num_classes] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be passed to the loss or postprocess functions.
        """
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]
       	with tf.name_scope('conv1_1') as scope:
		   net = tf.contrib.layers.conv2d(inputs=preprocessed_inputs,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu,scope=scope)
       
       	with tf.name_scope('conv1_2') as scope:
		   net = tf.contrib.layers.conv2d(inputs=preprocessed_inputs,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu,scope=scope)
       
       	with tf.name_scope('conv1_1') as scope:
		   net = tf.contrib.layers.conv2d(inputs=preprocessed_inputs,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu,scope=scope)
       
       	with tf.name_scope('conv1_1') as scope:
		   net = tf.contrib.layers.conv2d(inputs=preprocessed_inputs,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu,scope=scope)
       
       	with tf.name_scope('conv1_1') as scope:
		   net = tf.contrib.layers.conv2d(inputs=preprocessed_inputs,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu,scope=scope)
       
        net = tf.add(tf.matmul(net, fc9_weights), fc9_biases)
        prediction_dict = {'logits': net}
        return prediction_dict

    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations of specified models.
        
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        postprecessed_dict = {'classes': classes}
        return postprecessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
        prediction_dict: A dictionary holding prediction tensors.
        groundtruth_lists: A list of tensors holding groundtruth
        information, with one entry for each image in the batch.
        
        Returns:
        A dictionary mapping strings (loss names) to scalar tensors
        representing loss values.
        """
        logits = prediction_dict['logits']
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5),tf.trainable_variables())
        loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=groundtruth_lists)) + reg
        loss_dict = {'loss': loss}
        return loss_dict

