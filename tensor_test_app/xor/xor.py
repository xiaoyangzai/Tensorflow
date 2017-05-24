#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def add_layer(inputs_data,input_size,output_size,active_function = None,label = '1'):
	lay_label = 'layer'+label
	w = tf.Variable(tf.truncated_normal([input_size,output_size]),name = lay_label + 'weights')
	b = tf.Variable(tf.zeros([output_size]),name = lay_label + 'biases')

	w_plus_b = tf.add(tf.matmul(inputs_data,w),b)
	if active_function == None:
		return w_plus_b
	else:
		return active_function(w_plus_b)
	

def first_function():
	x_data =np.array([[1,1],[1,0],[0,1],[0,0]])
	y_data =np.array([[1,0],[0,1],[0,1],[1,0]])
	
	x_i = tf.placeholder(tf.float32,[None,2])
	y_data_i = tf.placeholder(tf.float32,[None,2])

	a_1 = add_layer(x_i,2,2,tf.nn.sigmoid,'1')
	a_2 = add_layer(a_1,2,2,tf.nn.softmax,'2')

	diff = -tf.reduce_sum(y_data_i*tf.log(a_2))	
	#diff = tf.reduce_sum(tf.square(y_data_i - a_2))	
	loss = tf.reduce_mean(diff)
	
	optimizer = tf.train.GradientDescentOptimizer(0.1)
	train = optimizer.minimize(loss)
	
	init = tf.initialize_all_variables()
	
	sess = tf.Session()
	sess.run(init)
	saver = tf.train.Saver()
	saver.restore(sess,'mode.ckpt')
	print (x_data[0],y_data[0],sess.run(a_2,feed_dict={x_i:[x_data[0]]}))
	print (x_data[1],y_data[1],sess.run(a_2,feed_dict={x_i:[x_data[1]]}))
	print (x_data[2],y_data[2],sess.run(a_2,feed_dict={x_i:[x_data[2]]}))
	print (x_data[3],y_data[3],sess.run(a_2,feed_dict={x_i:[x_data[3]]}))
	for step in range(3000):
		sess.run(train,feed_dict={x_i:x_data,y_data_i:y_data})
	
	print (x_data[0],y_data[0],sess.run(a_2,feed_dict={x_i:[x_data[0]]}))
	print (x_data[1],y_data[1],sess.run(a_2,feed_dict={x_i:[x_data[1]]}))
	print (x_data[2],y_data[2],sess.run(a_2,feed_dict={x_i:[x_data[2]]}))
	print (x_data[3],y_data[3],sess.run(a_2,feed_dict={x_i:[x_data[3]]}))
	saver.save(sess,"mode.ckpt")
	return

def main():
	first_function()
	return

if __name__ == '__main__':
	main()
