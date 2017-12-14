import tensorflow as tf
import numpy as np;
import cv2;
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import xml.etree.ElementTree as ET
import pickle

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
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32,shape=(None,image_shape[0],image_shape[1],image_shape[2]),name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32,name="keep_prob")

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    filter_ht = conv_ksize[0]
    filter_wh = conv_ksize[1]
    input_depth = x_tensor.get_shape().as_list()[3]
    
    weights = tf.Variable(tf.truncated_normal([filter_ht,filter_wh,input_depth,conv_num_outputs],stddev=5e-2))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    conv_stride_ht=conv_strides[0]
    conv_stride_wh = conv_strides[1]
    conv_stride=[1,conv_stride_ht,conv_stride_wh,1]
    
    p_size =[1,pool_ksize[0],pool_ksize[1],1]
    p_stride=[1,pool_strides[0],pool_strides[1],1]
    conv_layer = tf.nn.conv2d(x_tensor,weights,conv_stride,padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer,bias)
    #conv_layer = tf.nn.relu(conv_layer)
    #conv_layer = tf.nn.max_pool(conv_layer,p_size,p_stride,padding='SAME')
    return conv_layer 

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    shape=x_tensor.get_shape().as_list()
    new_dim = np.prod(shape[1:])
    flat_tensor = tf.reshape(x_tensor,[-1,new_dim])
    return flat_tensor
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv = conv2d_maxpool(x, 64, (5,5), (1,1), (3,3), (2,2))
    #conv = conv2d_maxpool(conv, 64, (5,5), (1,1), (3,3), (2,2))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    #flatconv = flatten(conv)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    #fc = fully_conn(flatconv, 1000)
    #fc = fully_conn(fc, 500)
    #fc = tf.nn.dropout(fc, keep_prob)
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    #out = output(fc, 10)
    out = conv2d_maxpool(conv, 3, (5,5), (1,1), (3,3), (2,2))
    
    # TODO: return output
    return out

tf.reset_default_graph()

# Inputs
x = neural_net_image_input((500, 500, 3))
y = neural_net_label_input((500, 500, 3))
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.square(flatten(logits)-flatten(y)))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(flatten(logits), flatten(y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer,feed_dict={x:feature_batch,y:label_batch,keep_prob:keep_probability})

def load_preprocess_training_batch(batch_id):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'pre_processed_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    features_array = np.zeros((len(features),500,500,3))
    labels_array = np.zeros((len(labels),500,500,3))
    i=0
    for b in features:
        features_array[i] = b
        i=i+1
    features_array = features_array.astype(np.float32)
    i=0
    for b in labels:
        labels_array[i] = b
        i=i+1
    labels_array = labels_array.astype(np.float32)
    # Return the training data in batches of size <batch_size> or less
    return features_array, labels_array
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost,feed_dict={x:feature_batch,y:label_batch,keep_prob:1.0})
    vaccuracy = session.run(accuracy,feed_dict={x:valid_features,y:valid_labels,keep_prob:1.0})
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss,vaccuracy))
epochs = 10
batch_size = 256
keep_probability = 0.75

valid_features, valid_labels = load_preprocess_training_batch(1)
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    save_model_path = './capstone'
    # Training cycle
    for epoch in range(epochs):
        batch_i = 0
        #for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
        batch_features, batch_labels = load_preprocess_training_batch(batch_i)
 
        #print(batch_labels_array[1].shape)
        #print("blablablablablabla")
        #print(batch_features)
        
        
        train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
