#!/usr/bin/python
#-*- coding:utf-8 -*-
import cv2
import sys
import glob
import numpy as np
import os
import tensorflow as tf
def get_test_data(images_path):
	images = []
	labels = []
	images_path = os.path.join(images_path,"*.jpg")
	count = 0
	for image_file in glob.glob(images_path):
		count += 1
		print('Load {} images.'.format(count))
		image = cv2.imread(image_file)
		image = cv2.imread(image_file)
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		label = int(image_file.split('_')[-1].split('.')[0])
		images.append(image)
		labels.append(label)
	images = np.array(images)
	labels = np.array(labels)
	return images,labels
def main():
	print "load model..."
	if len(sys.argv) != 4:
		print"[test images path] [model meta path] [model path]"
		return
	test_images_path = sys.argv[1]
	model_checkpoint_file = sys.argv[2]
	model_checkpoint_path = sys.argv[3]
	print "load test images ..."
	test_images,test_labels = get_test_data(test_images_path)
	print "load test images ok!"
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_checkpoint_file)
		saver.restore(sess,tf.train.latest_checkpoint(model_checkpoint_path))
		print "restore model ok!"
		graph = tf.get_default_graph()
		inputs = graph.get_operation_by_name('inputs').outputs[0]
		labels = graph.get_operation_by_name('labels').outputs[0]
		accuracy = tf.get_collection("predict_network")[0]
		classes = tf.get_collection("classes")[0]
		probs = tf.get_collection("probs")[0]
		acr,cls,prbs = sess.run([accuracy,classes,probs],feed_dict={inputs:test_images,labels:test_labels})
        print "Accuracy: ",acr
        for index in range(len(cls)):
            print "%d : %.3f"%(cls[index],prbs[index][cls[index]])

if __name__ == "__main__":
	main()
