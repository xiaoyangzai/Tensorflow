#!/usr/bin/python
#-*- coding: utf-8 -*-

import os 
from glob import glob
import cv2

def create_images_labels_dict(dict_file):
    s = open(dict_file)
    images_labels = {}
    for line in s:
        image = line.strip().split(" ")[0]
        label = int(line.strip().split(" ")[1])
        #print("image: %s - label: %d"%(image,label))
        images_labels[image] = label
    return images_labels

def rename_images_labels(images_path,images_labels_dict,label_text = "for_train"):
    images_orignal_path = glob(images_path+"/*.jpg")
    index = 0
    for item in images_orignal_path:
        image_name = os.path.basename(item)
        image_label = images_labels_dict[image_name]
        image_newname = os.path.dirname(item) + "/image_for_train_%d"%index + "-%d.jpg"%image_label
        os.rename(item,image_newname)
        index += 1
    print("Rename image name ok!!")
    return

def resize_images(images_path,new_path,newsize):
    images = glob(images_path+"/*.jpg")
    for item in images:
        tmp = os.path.basename(item).split('.')
        tmpid = int(tmp[1])
        print tmpid
        if tmpid > 3000:
            continue
        image = cv2.imread(item)
        if tmp[0] == "cat":
            newname = os.path.join(new_path,tmp[0] + tmp[1]+"-0.jpg") 
        else:
            newname = os.path.join(new_path,tmp[0] + tmp[1]+"-1.jpg") 
        newimage = cv2.resize(image,newsize)
        cv2.imwrite(newname,newimage)
    print("resize ok!!") 
    return


def main():
    #images_path = glob("./dir/*.jpg")
    #print images_path
    #images_labels = create_images_labels_dict('train.txt')
    #rename_images_labels('train/',images_labels,"for_test")
    resize_images('train','catdog_dataset',(64,64))
    return

if __name__ == "__main__":
    main()

