import tensorflow as tf
from six.moves import cPickle


# Loading net skeleton with parameters name and shapes.
with open("util/net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [2, 2, 2],
                 [12], 
                 [1], 
                 [1]]
n_classes = 21
ks = 3


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
class SegmentationModel(object):
    def __init__(self, weights_path=None):
        self.variables = self._create_variables(weights_path)
    
    def _create_variables(self, weights_path):
        var = list()
        index = 0
        
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f) # Load pre-trained weights.
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
                if "/w" in name: # Weight filter.
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var
    
    def _create_network(self, input_batch, keep_prob):
        
        current = input_batch
        v_idx = 0
        for b_idx in range(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
                
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current
    
    
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        
        raw_output = self._create_network(tf.cast(input_batch, tf.float32),  keep_prob=tf.constant(1.0))
        #print(raw_output.shape)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        #print(raw_output.shape)
        raw_output = tf.argmax(raw_output, axis=3)
        #print(raw_output.shape)
        raw_output = tf.expand_dims(raw_output, dim=3) # Create 4D-tensor.
        #print(raw_output.shape)
        return tf.cast(raw_output, tf.uint8)
    
    def prepare_label(self,input_batch,new_size):
        with tf.name_scope('label_encode'):
            input_batch= tf.image.resize_nearest_neighbor(input_batch,new_size)
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=21)
        return input_batch
    
    def loss(self, img_batch, label_batch,keep_prob_input):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), keep_prob_input)
        #print(raw_output.shape)
        prediction = tf.reshape(raw_output, [-1, n_classes])
        label_batch = self.prepare_label(label_batch,tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        #print(label_batch.shape)
        # Pixel-wise softmax loss.
        losssce = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(losssce)
        return reduced_loss