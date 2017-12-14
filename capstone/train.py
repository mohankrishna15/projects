import tensorflow as tf
from six.moves import cPickle
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from model import *

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
        filename = 'preprocessed_data/train_data/pre_processed_batch_' + str(batch_id) + '.p'
        loaded_features, loaded_labels = cPickle.load(open(filename, mode='rb'))
        # Return the training data in batches of size <batch_size> or less
        return loaded_features, loaded_labels


epochs = 50
keep_probability = 0.75
save_model_path = "./capstone"
image_save_path = "images/"
tf.reset_default_graph()
# Inputs
x = neural_net_image_input((500, 500, 3))
y = neural_net_label_input((500, 500, 1))
keep_prob = neural_net_keep_prob_input()

net = SegmentationModel(None)
cost = net.loss(x, y,keep_prob)
cost_summary=tf.summary.scalar("loss",cost)
optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
saver = tf.train.Saver(max_to_keep=40);  
pred = net.preds(x)

sess = tf.Session()
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/home/systems/capstone/logs/', sess.graph)
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    for batch_i  in range(0, 1):
        start_time = time.time()
        batch_features, batch_labels = load_preprocess_training_batch(batch_i)
        f = np.zeros([16,500,500,3])
        l = np.zeros([16,500,500,1])
        for i in range(16):
            f[i] = batch_features[i,:,:,:]
            l[i] = batch_labels[i,:,:,:]
        loss_value,pred_value,_=sess.run([cost,pred,optim],feed_dict={x:f,y:l,keep_prob:keep_probability})
        fig, axes = plt.subplots(2, 3, figsize = (16, 12))
        for i in range(2):
            axes.flat[i * 3].set_title('data')
            axes.flat[i * 3].imshow((f[i])[:, :, ::-1].astype(np.uint8))

            axes.flat[i * 3 + 1].set_title('mask')
            axes.flat[i * 3 + 1].imshow(l[i, :, :, 0])

            axes.flat[i * 3 + 2].set_title('pred')
            axes.flat[i * 3 + 2].imshow(decode_labels(pred_value[i, :, :, 0]))
        plt.savefig(image_save_path + str(start_time) + ".png")
        plt.close(fig)
        duration = time.time() - start_time
        print('step {:d}, batch {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(epoch, batch_i, loss_value, duration)) 
    saver.save(sess, save_model_path)
