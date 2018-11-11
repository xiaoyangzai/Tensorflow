#!/usr/bin/python

import tensorflow as tf
import sys
import numpy as np
import time
import math
import matplotlib.pyplot as plt
def draw_result(steps,losses,accuracies,test_accuracies):
	fig,ax = plt.subplots(2,1)
	ax[0].plot(steps,losses,'-')
	ax[0].set_title('Loss of each step')
	ax[0].set_xlabel('Step')
	ax[0].set_ylabel('Loss')

	ax[1].plot(steps,accuracies,'b-')
	ax[1].plot(steps,test_accuracies,'r-')
	ax[1].set_title('Accuracy of each step')
	ax[1].set_xlabel('Step')
	ax[1].set_ylabel('Accuracy')
	plt.show()

def add_layer(inputs_data,input_size,output_size,active_function = None,label = '1'):
	lay_label = 'layer'+label
	w = tf.Variable(tf.truncated_normal([input_size,output_size]),name = lay_label + 'weights')
	b = tf.Variable(tf.zeros([output_size]),name = lay_label + 'biases')

	w_plus_b = tf.add(tf.matmul(inputs_data,w),b)
	if active_function == None:
		return w_plus_b
	else:
		return active_function(w_plus_b)

def one_hot(label,num_classes):
	onehot_label = [0 for i in range(num_classes)]
	onehot_label[label - 1] = 1
	return onehot_label

def load_dataset(dataset):
	fd = open(dataset)
	x_data = []
	y_data = []
	lines = fd.readlines()
	for line in lines:
		if line[0] == '#':
			continue
		line = line.strip().split(',')
		if(int(line[-1]) > 3):
			continue
		x = []
		for item in line[:-1]:
			x.append(float(item))	
		x_data.append(x)
		y_data.append(one_hot(int(line[-1]),3))
	fd.close()
	x_data = np.array(x_data)
	y_data = np.array(y_data)
	x_max,x_min= x_data.max(),x_data.min()
	x_data = (x_data - x_min)/(x_max - x_min)
	y_max,y_min= y_data.max(),y_data.min()
	y_data = (y_data - y_min)/(y_max - y_min)
	return x_data,y_data

def first_function():
	x_data,y_data = load_dataset(sys.argv[1])
	test_x_data,test_y_data = load_dataset(sys.argv[2])
	inputs = tf.placeholder(tf.float32,shape=[None,len(x_data[0])],name="inputs")
	labels = tf.placeholder(tf.float32,shape=[None,3],name='labels')
	
	w = tf.Variable(tf.random_uniform([len(x_data[0]),3]))
	b = tf.Variable(tf.zeros([3]))
	y = tf.nn.relu(tf.matmul(inputs,w)+b)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=y))
	optimizer = tf.train.GradientDescentOptimizer(0.1)
	train = optimizer.minimize(loss)
	
	init = tf.initialize_all_variables()
	
	sess = tf.Session()
	sess.run(init)
	saver = tf.train.Saver()
	classes = tf.cast(tf.argmax(y,axis=1),dtype=tf.int32)
	y_label = tf.cast(tf.argmax(labels,axis=1),dtype=tf.int32)
	tf.add_to_collection("classes",classes)
	acc = tf.reduce_mean(tf.cast(tf.equal(classes,y_label),'float'))
	tf.add_to_collection("accuracy",acc)
	logfile = "train.log"
	logfd = open(logfile,'w')
	print "begin to train model...."
	logfd.write("#step,loss,accuracy\n")
	steps = []
	losses = []
	accuracies = []
	test_accuracies = []
	if len(sys.argv) == 4:
		saver.restore(sess,tf.train.latest_checkpoint(sys.argv[3]))
		print "# model restore ok!"
	for step in range(5000):
		steps.append(step)
		index = np.random.choice(len(x_data),50)
		batch_x_data = x_data[index]
		batch_y_data = y_data[index]

		sess.run(train,feed_dict={inputs:batch_x_data,labels:batch_y_data})
		loss_,acc_ = sess.run([loss,acc],feed_dict={inputs:batch_x_data,labels:batch_y_data})
		test_acc_ = sess.run(acc,feed_dict={inputs:test_x_data,labels:test_y_data})
		losses.append(loss_)
		accuracies.append(acc_)
		test_accuracies.append(test_acc_)
		print "step[%d]: ,loss: %f,accuracy: %.2f,test accuracy: %.2f"%(step,loss_,acc_,test_acc_)
		if step%100 == 0:
			saver.save(sess,"checkpoint/mode.ckpt")
			#time.sleep(1)
		logfd.write("%d,%.3f,%.3f\n"%(step,loss_,acc_))
	logfd.close()
	print "Training finished!"
	draw_result(steps,losses,accuracies,test_accuracies)
	return

def main():
	first_function()
	return

if __name__ == '__main__':
	main()
