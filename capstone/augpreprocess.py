import numpy as np
import xml.etree.ElementTree as ET
import pickle
from scipy import misc
import glob
from PIL import Image
import tensorflow as tf
from six.moves import cPickle

path = "/augmented_dataset/VOC_aug/dataset/"
im_sz = 500
batch_size=16
with open(path+"train.txt") as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
train_names = [x.strip() for x in content]

features = np.array([misc.imread(path+"img/"+x+".jpg") for x in train_names])
output = np.array([misc.imread(path+"cls_png/"+x+".png") for x in train_names])

with open(path+"val.txt") as f:
    contentval = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
val_names = [x.strip() for x in contentval]
features_val = np.array([misc.imread(path+"img/"+x+".jpg") for x in val_names])
output_val = np.array([misc.imread(path+"cls_png/"+x+".png") for x in val_names])
IMG_MEAN = np.array((116.66876762,104.00698793,122.67891434), dtype=np.float32)
def encode_labels(label_batch):
        colormap = {(0,0,0):0, (128,0,0):1, (0,128,0):2, (128,128,0):3, (0,0,128):4, (128,0,128):5, (0,128,128):6, (128,128,128):7, (64,0,0):8, (192,0,0):9, (64,128,0):10, (192,128,0):11, (64,0,128):12, (192,0,128):13, 
            (64,128,128):14, (192,128,128):15, (0,64,0):16, (128,64,0):17, (0,192,0):18, (128,192,0):19, (0,64,128):20}                                            
        gndTruth = np.zeros((label_batch.shape[0],500,500,1), dtype=np.int)
        for i in range(label_batch.shape[0]):
            for j in range(500):
                for k in range(500):   
                    if(colormap.get(tuple(label_batch[i][j,k]))):
                        gndTruth[i,j,k]=colormap.get(tuple(label_batch[i][j,k]))
                    else:
                        gndTruth[i,j,k] = 0
        return gndTruth
    
def preprocess_data(sess,img,lb):
    img_tensor = tf.image.resize_image_with_crop_or_pad(img, 500, 500)
    labels_tensor = tf.image.resize_image_with_crop_or_pad(lb, 500, 500)
    img_batch,lbls = sess.run([img_tensor,labels_tensor])
    img_batch[:,:,:] = img_batch[:,:,:] - IMG_MEAN
    return img_batch,lbls
    
sess = tf.Session()
total_features = np.zeros((16,500,500,3))
total_output = np.zeros((16,500,500,1))

total_features_val = np.zeros((16,500,500,3))
total_output_val = np.zeros((16,500,500,3))

batch_size=16
num_train_batches = int(len(features)/batch_size)
num_val_batches = int(len(features_val)/batch_size)

for i in range(0,num_train_batches+1):
    for j in range(0,16):
        total_features[j],total_output[j] = preprocess_data(sess,features[int(i*batch_size)+j],np.expand_dims(output[int(i*batch_size)+j],2))
    pickle.dump((total_features, total_output), open("aug_preprocessed_data/train_data/pre_processed_batch_"+str(i)+".p", 'wb'))

for i in range(0,num_val_batches+1):
    for j in range(0,16):
        total_features_val[j],total_output_val[j] = preprocess_data(sess,features_val[int(i*batch_size)+j],np.expand_dims(output_val[int(i*batch_size)+j],2))
    pickle.dump((total_features_val, total_output_val), open("aug_preprocessed_data/val_data/pre_processed_batch_"+str(i)+".p", 'wb'))
