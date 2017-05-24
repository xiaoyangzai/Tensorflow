#!/usr/bin/python

import tensorflow as tf
import numpy as np
def purline(input_data,weights,bias):
	return weights*input_data + bias

def first_function():
	x_data = np.random.rand(100).astype("float32")
	y_data = x_data * 0.1 + 0.3
	
	w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
	b = tf.Variable(tf.zeros([1]))
	y = purline(x_data,w,b)
	
	loss = tf.reduce_mean(tf.square(y - y_data))
	
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = optimizer.minimize(loss)
	
	init = tf.initialize_all_variables()
	
	sess = tf.Session()
	sess.run(init)

	for step in range(100):
		sess.run(train)
		print sess.run(y)
		print(sess.run(loss),sess.run(w),sess.run(b))
	return
def main():
	first_function()
	return

if __name__ == '__main__':
	main()
