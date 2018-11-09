#!/usr/bin/python

#-*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

from captcha.image import ImageCaptcha

def generate_captcha(text='1'):
    '''Generate a digit image'''
    capt = ImageCaptcha(width=64,height=64,font_sizes=[32])
    image = capt.generate_image(text)
    image = np.array(image,dtype=np.uint8)
    return image

if __name__ == "__main__":
    output_dir = './datasets/images/'
    for i in range(20000):
        label = np.random.randint(0,10)
        image = generate_captcha(str(label))
        image_name = 'image{}_{}.jpg'.format(i+1,label)
        output_path = output_dir + image_name
        cv2.imwrite(output_path,image)
        sys.stdout.write(str(i)+"-"+str(10000) + '\r')
        sys.stdout.flush()
