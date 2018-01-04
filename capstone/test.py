import tensorflow as tf
from six.moves import cPickle
import numpy as np
import time
from matplotlib import pyplot as plt
from PIL import Image
from model import *

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation 
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    tf.summary.histogram(name, variable)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    tf.summary.histogram(name, variable)
    return variable


def neural_net_image_input(image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        # TODO: Implement Function
        return tf.placeholder(tf.float32,shape=(None,image_shape[0],image_shape[1],image_shape[2]),name="x")
def neural_net_label_input(image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        # TODO: Implement Function
        return tf.placeholder(tf.uint8,shape=(None,image_shape[0],image_shape[1],image_shape[2]),name="y")

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32,name="keep_prob")



    
    
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    
def decode_labels(mask):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    imgrgb = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = imgrgb.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 21:
                pixels[k_,j_] = label_colours[k]
    return np.array(imgrgb)
def encode_labels(label_batch):
        colormap = {(0,0,0):0, (128,0,0):1, (0,128,0):2, (128,128,0):3, (0,0,128):4, (128,0,128):5, (0,128,128):6, (128,128,128):7, (64,0,0):8, (192,0,0):9, (64,128,0):10, (192,128,0):11, (64,0,128):12, (192,0,128):13, 
            (64,128,128):14, (192,128,128):15, (0,64,0):16, (128,64,0):17, (0,192,0):18, (128,192,0):19, (0,64,128):20}                                            
        gndTruth = np.zeros((label_batch.shape[0],label_batch[0].shape[0],label_batch[0].shape[1],1), dtype=np.int)
        for i in range(gndTruth.shape[0]):
            for j in range(gndTruth.shape[1]):
                for k in range(gndTruth.shape[2]):   
                    if(colormap.get(tuple(label_batch[i][j,k]))):
                        gndTruth[i,j,k]=colormap.get(tuple(label_batch[i][j,k]))
                    else:
                        gndTruth[i,j,k] = 0
        return gndTruth
def load_preprocess_training_batch(batch_id):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with tf.name_scope("create_inputs"):
        filename = 'preprocessed_data/val_data/pre_processed_batch_' + str(batch_id) + '.p'
        loaded_features, loaded_labels = cPickle.load(open(filename, mode='rb'))
        # Return the training data in batches of size <batch_size> or less
        return loaded_features.astype(np.int64), loaded_labels.astype(np.int64)
    
    
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


total_batches=90;
tf.reset_default_graph()

net = SegmentationModel(None)

x = neural_net_image_input((500, 500, 3))
y = neural_net_label_input((500, 500, 1))
keep_prob = neural_net_keep_prob_input()

pred = net.preds(x)

predconv = tf.to_int32(pred, name='ToInt32')
yconv = tf.to_int32(y, name='ToInt32') 
mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(predconv,yconv, num_classes=21)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.initialize_all_variables()
    
sess.run(init)
sess.run(tf.initialize_local_variables())

saver = tf.train.Saver()

saver.restore(sess, "./aug_capstone")


for batch_i in range(total_batches):
    batch_features, batch_labels = load_preprocess_training_batch(batch_i)
    test_input=np.zeros([1,500,500,3])
    for img_i in range(16):
        test_input[0,:,:,:] = batch_features[img_i,:,:,:]

        lbl_img = np.zeros([1,500,500,1],dtype=np.int)
        for i in range(0,500):
            for j in range(0,500):
                lbl_img[0,i,j,0] = int(batch_labels[img_i,i,j,0])

        pred_value,up = sess.run([pred,update_op],feed_dict={x:test_input,y:lbl_img})
    print('batch: '+ str(batch_i))

print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
