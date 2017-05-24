#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def pureline(x):
	print x.shape
	return x
def add_layer(inputs,in_size,out_size,activation_function = None,label = '1'):
	#add layer and return the output of this layer
	with tf.name_scope('layer'+label):
		with tf.name_scope('weights'):
			w = tf.Variable(tf.random_normal([in_size,out_size]),name='W'+label)
		with tf.name_scope('biases'):
			b = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b'+label)

		with tf.name_scope('w_plus_b'+label):
			w_plus_b = tf.add(tf.matmul(inputs,w),b)

		if activation_function == None:
			return w_plus_b
		else:
			return activation_function(w_plus_b)


def first_function():
	x_data = np.linspace(-2,2,100)[:,np.newaxis]
	y_data = []
	for x in x_data:
		y = []
		y.append(1 + math.sin(x[0]  * math.pi/4))
		y_data.append(y)
	
	with tf.name_scope('input'):
		x_i = tf.placeholder(tf.float32,[None,1],name='x_in')
		y_data_i = tf.placeholder(tf.float32,[None,1],name='y_in')
	
	
	a_1 = add_layer(x_i,1,2,activation_function=tf.sigmoid) 

	a_2 = add_layer(a_1,2,1,label='2',activation_function=pureline) 


	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.square(a_2 - y_data_i))
	
	with tf.name_scope('train'):
		optimizer = tf.train.GradientDescentOptimizer(0.1)
		train = optimizer.minimize(loss)
	
	init = tf.initialize_all_variables()

	sess = tf.Session()

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data,y_data)
	plt.ion()
	plt.show()

	writer = tf.summary.FileWriter("logs/",sess.graph)

	sess.run(init)
	saver = tf.train.Saver()
	try:
		saver.restore(sess,"mode.ckpt")
	except Exception:
		pass

	for step in range(2000):
		sess.run(train,feed_dict={x_i:x_data,y_data_i:y_data})
		if step % 50 == 0:
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			temp = sess.run(a_2,feed_dict={x_i:x_data})
			
			lines = ax.plot(x_data,temp,'r-',lw=5)
			plt.pause(1)
	saver.save(sess,"mode.ckpt")
	return

def main():
	first_function()
	return

if __name__ == '__main__':
	main()
