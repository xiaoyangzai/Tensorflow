#!/usr/bin/python
#-*- coding:utf-8 -*-

import cv2
import sys
import glob
import numpy as np
import os
import tensorflow as tf
import model

flags = tf.app.flags
flags.DEFINE_string('images_path',None,'Path to training images.')
flags.DEFINE_string('model_output_path',None,'Path to model checkpoint')
FLAGS = flags.FLAGS


def get_train_data(images_path):
    '''Get the training images from images_path

    Args:
        images_path: Path to the training images.
    
    Returns:
        images: A list of images.
        labels: A list of integers representing teh classes of images.

    Raise:
        ValueError: If images_path is not exist.
    '''
    if not os.path.exists(images_path):
        raise ValueError('Images path is not exist.')

    images = []
    labels = []
    images_path = os.path.join(images_path,"*.jpg")
    count = 0
    for image_file in glob.glob(images_path):
        count += 1
        if count % 100 == 0:
            print('Load {} images.'.format(count))

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = int(image_file.split('-')[-1].split('.')[0])
        images.append(image)
        labels.append(label)
        print labels[:10]
    images = np.array(images)
    labels = np.array(labels)
    train_images = images[:len(images) - 50]
    train_labels = labels[:len(labels) - 50]
    test_images = images[len(images) - 50:]
    test_labels = labels[len(labels) - 50:]
    return train_images, train_labels,test_images,test_labels

def next_batch_set(images,labels,batch_size = 100):
    '''
        Generate a batch training data.

    Args:
        images: A 4-D array representing the training images.
        labels: A 1-D array representing the classes of the training images.
        batch_size : An integer.

    Returns:
        batch_images: A batch of images. 
        batch_labels: A batch of labels.
    '''
    indices = np.random.choice(len(images),batch_size)
    batch_images = images[indices]
    batch_labels = labels[indices]
    return batch_images,batch_labels

def main(_):
    if len(sys.argv) < 3:
        return
    inputs = tf.placeholder(tf.float32,shape=[None,64,64,3],name="inputs")
    labels = tf.placeholder(tf.int32,shape=[None],name='labels')

    print("Build the claser Model...")
    cls_model = model.Model(is_training=True,num_classes=2)

    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict,labels)
    loss = loss_dict['loss']

    postprocessed_dict = cls_model.postprocess(prediction_dict)

    classes = postprocessed_dict['classes']
    classes_ = tf.identity(classes, name="classes")

    acc = tf.reduce_mean(tf.cast(tf.equal(classes,labels),'float'))
    tf.add_to_collection("predict_network",acc)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.9)

    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
    train_step = optimizer.minimize(loss,global_step)
    saver = tf.train.Saver()
    train_images,train_labels,test_images,test_labels = get_train_data(sys.argv[1])

    init = tf.global_variables_initializer()
    log = open("train.log","w")
    with tf.Session() as sess:
        sess.run(init)
        if len(sys.argv) == 4:
            saver.restore(sess,tf.train.latest_checkpoint(sys.argv[3])) 
            print "#sucessfully load model from %s"%sys.argv[3]

        for i in range(2000):
            batch_images,batch_labels = next_batch_set(train_images,train_labels)
            train_dict = {inputs:batch_images,labels: batch_labels}

            sess.run(train_step, feed_dict=train_dict)

            loss_, acc_ = sess.run([loss,acc],feed_dict=train_dict)
            test_dict = {inputs:test_images,labels: test_labels}
            test_acc_ = sess.run(acc,feed_dict=test_dict)
            train_text = 'step:{},loss:{},train_acc:{},test_acc:{}'.format(i+1,loss_,acc_,test_acc_) 
            print(train_text)
            log.write(train_text+"\n")
            if i % 10 == 0:
                saver.save(sess,sys.argv[2]+"model_checkpoint.pkt")
        log.close()

if __name__ == '__main__':
    tf.app.run()
