#!/usr/bin/python

import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt
def draw_result(steps,losses,accuracies):
	fig,ax = plt.subplots(2,1)
	ax[0].plot(steps,losses,'-o')
	ax[0].set_title('Loss of each step')
	ax[0].set_xlabel('Step')
	ax[0].set_ylabel('Loss')

	ax[1].plot(steps,accuracies,'-x')
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
	optimizer = tf.train.GradientDescentOptimizer(0.05)
	train = optimizer.minimize(loss)
	
	init = tf.initialize_all_variables()
	
	sess = tf.Session()
	sess.run(init)
	saver = tf.train.Saver()
	tf.add_to_collection("probs",a_2)
	classes = tf.cast(tf.argmax(a_2,axis=1),dtype=tf.int32)
	y_label = tf.cast(tf.argmax(y_data_i,axis=1),dtype=tf.int32)
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
	for step in range(3000):
		#time.sleep(1)
		steps.append(step)
		sess.run(train,feed_dict={x_i:x_data,y_data_i:y_data})
		loss_,acc_ = sess.run([loss,acc],feed_dict={x_i:x_data,y_data_i:y_data})
		losses.append(loss_)
		accuracies.append(acc_)
		print "step[%d]: ,loss: %f,accuracy: %.2f"%(step,loss_,acc_)
		if step%10 == 0:
			saver.save(sess,"checkpoint/mode.ckpt")
		logfd.write("%d,%.3f,%.3f\n"%(step,loss_,acc_))
	logfd.close()
	print "Training finished!"
	draw_result(steps,losses,accuracies)
	return

def main():
	first_function()
	return

if __name__ == '__main__':
	main()
