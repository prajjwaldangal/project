import tensorflow as tf
import numpy as np
import random

import algorithm_lib as alib
import plot_lib as plt2

""""
Our network uses a combination of pixel intensities and weights.

The neural network specifications:
Zero padding = 1 on each edge
Stride = 2 both horizontally and vertically
    
    1x32x32 ----------> 32x16x16 --------> 32x8x8 ----------> 16x4x4 --------> 16x2x2 ---------> 8x1x1
    input     Conv. 3x3            MP             Conv. 2x2          MP               Conv. 2x2  
              st=2, 32             2x2            st=2, 16           2x2              st=2, 8
              filters                               filters                           filters
              
              
    
    
    8x1x1 ----------> 10N ---------> 4N -----------> 1N (class prediction)
          fc layer         fc layer      fc layer
    
# output volume formula:  (Img width−Filter size+2*Padding)/Stride+1

30x30 input images, 3  conv. layer, 2 max. pool layers, 3 fully-conn. layers
#relu after conv layer
loss function = SVM loss
activation function = relu

"""

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 2
num_filters2 = 16

# Convolutional Layer 3.
filter_size3 = 2
num_filters3 = 8

# Fully-connected layer.
fc1_size = 10             # Number of neurons in the first fully-connected layer.
fc2_size = 7              # second fully_connected layer

# Number of color channels for the images: 1 channel for gray-scale, 3 for RGB.
num_channels = 1

# image dimensions (only squares for now)
#img_size = 128
img_size = 30

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
# classes = ['3c', '4a', '4b', '4c']
classes = ['4a']
num_classes = len(classes)

# batch size
BATCH_SIZE = 10

# validation split
validation_size = .14

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'data/train/' # + '3c'/'4a'/'4b' ...
test_path = 'data/test/' # similar to train path
checkpoint_dir = "models/" #

################################ tensors in tensorflow  ###############################################################
# Rank 0 tensors: tf.Variable("Elephant", tf.string), tf.Variable(451/3.14159/2.2+7j, tf.int16, tf.float64,
#                                                                                     tf.complex64) respectively
# Rank 1 tensors: tf.Variable(["Elephant"]/[3.1416, 2.7183]/[2,3,4]/[12+7j, 2-3.5j],
#                                                           tf.string/tf.float32/tf.int32/tf.complex64) respectively
# similarly rank 2 would be list of list, rank 3 would be list of list of list

# a rank 2 tensor with shape [3,4] : [ [1, 2, 3, 4], [2, 4, 6, 8], [-1, -2, -3, -4] ]
# following is a rank 3 tensor with shape [1, 4, 3]:
# [ [
#       [ 1, 2, 3 ],
#       [ 2, 4, 6 ],
#       [ 3, 6, 9 ],
#       [ 5, 10, 15 ]
#  ], [
#       [-1, -2, -3],
#       [-2, -4, -6],
#       [-3, -6, -9],
#       [-5, -10, -15]
#   ] ], more info: https://www.tensorflow.org/guide/tensors
rank_three_tensor = tf.ones([3,4,5])
rank_three_tensor_np = np.ones((3,4,5))
r_two_tf = tf.reshape(rank_three_tensor, [10,-1])
r_two_np = rank_three_tensor_np.reshape((10,6))

# initialize train x and y placeholders
X = tf.placeholder(tf.float32, (int(num_classes * BATCH_SIZE * (1-validation_size)), 30, 30, 1), name = 'X')
Y = tf.placeholder(tf.int8, (int(num_classes * BATCH_SIZE * (1-validation_size))), name='Y')


# the matrices to be learnt
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# the biases, also are learnt
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# we need to reduce the 4-dim tensor to 2-dim which can be used as 
# input to the fully-connected layer
def flatten_layer(layer):
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    # i.e. the first dimension will be num_of_elements_
    layer_flat = tf.reshape(layer, [-1, num_features])

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    print(input)
    # c_op inside,
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         #strides=[1, 1, 1, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

layer_conv1, weights_conv1 = \
    new_conv_layer(input=X,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_channels,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

#
# layer_conv3, weights_conv3 = \
#     new_conv_layer(input=layer_conv2,
#                    num_input_channels=num_channels,
#                    filter_size=filter_size3,
#                    num_filters=num_filters3,
#                    use_pooling=False)
# print(layer_conv3)

# layer_fc2 = new_fc_layer()
# tf.train.GradientDescentOptimizer vs tf.train.AdamOptimizer for mnist vs cnn respectively.
session = tf.Session()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
# Counter for total number of iterations performed so far.

total_iterations = 0
import time

# what is img_size_flat?
def fetch_data_single(hair_type, segmented_thus_less=True, dim=(img_size, img_size)):
    """

    :param segmented_thus_less: get segmented images or no
    :param dim: the dimension of input images
    :param hair_type:   hair_type [3c, 4a, 4b, 4c]
    :return:
    """
    if not segmented_thus_less:
        segmented = False
    else:
        segmented = True

    bin4a, _, _, _, _ = alib.load_preprocess_contours(hair_type, BATCH_SIZE, dim, segmented=segmented)
    # once multiple classes' training started, use arr = [bin3c, bin4a, ...]
    idx = int((1-validation_size)*len(bin4a))
    train_batch_x = bin4a[:idx]
    train_batch_y = np.zeros((len(train_batch_x)))
    valid_batch_x = bin4a[idx:]
    valid_batch_y = np.zeros((len(valid_batch_x)))
    feed_dict_train = {X: train_batch_x,
                       Y: train_batch_y}

    #feed_dict_validate = {X: valid_batch_x,
    #                      Y: valid_batch_y}
    session.run(optimizer, feed_dict=feed_dict_train)
    # Print status at end of each epoch (defined as full pass through training dataset).

# help(tf.keras)
# https://www.tensorflow.org/tutorials/deep_cnn
# vs
# https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb
# vs
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/#more-452
# vs
# https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks
# https://www.tensorflow.org/tutorials/images/image_recognition

def random_mix(ls, hair_types):
    """
    :param ls: list of different hair type images, num_classes x BATCH_SIZE x img_height x img_width
    :return: train and validation image sets along with their corresponding labels
    """
    if ls == [] or ls[0] == [] or ls[1] == []:
        return
    n_hair_types = len(ls)
    n_imgs = len(ls[0]) # number of images in each hair_type array
    img_height = len(ls[0][0])
    img_width = len(ls[0][0][0])
    # the following code turns a x b x c x d arr into l x c x d where l = a x b, mixes them and separates
    # into train and validation sets
    train_x = []
    valid_x = []
    train_y = []
    valid_y = []
    # arr = np.zeros((n_hair_types * n_imgs, img_height, img_width))
    for i, hair_matrices in enumerate(ls):
        l = int((1-validation_size)*n_imgs)+1
        # label=hair_types[i]
        for idx, img in enumerate(hair_matrices[:l]):
            train_x.append(img)
            train_y.append(i)
        for idx, img in enumerate(hair_matrices[l:]):
            valid_x.append(img)
            valid_y.append(i)
    # mix
    print("Shuffling...")
    alib.dots(5)
    t = len(train_x)
    v = len(valid_x)
    train_rtrn_x = np.zeros((t, img_height, img_width))
    train_rtrn_y = np.zeros((t))
    valid_rtrn_x = np.zeros((v, img_height, img_width))
    valid_rtrn_y = np.zeros((v))
    train_seen = {}
    valid_seen = {}
    train_cnt = 0
    valid_cnt = 0
    done = False
    while not done:
        train_rand_int = random.randint(0, t-1)
        valid_rand_int = random.randint(0, v-1)
        if not train_rand_int in train_seen:
            # insert delete x
            train_rtrn_x = np.insert(train_rtrn_x, train_rand_int, train_x[train_rand_int], 0)
            train_rtrn_x = np.delete(train_rtrn_x, train_rand_int+1, 0)
            # insert delete y
            train_rtrn_y = np.insert(train_rtrn_y, train_rand_int, train_y[train_rand_int], 0)
            train_rtrn_y = np.delete(train_rtrn_y, train_rand_int+1, 0)
            train_cnt += 1

        if not valid_rand_int in valid_seen:
            # same as above: insert delete x
            valid_rtrn_x = np.insert(valid_rtrn_x, valid_rand_int, valid_x[valid_rand_int], 0)
            valid_rtrn_x = np.delete(valid_rtrn_x, valid_rand_int+1, 0)
            # insert delete y
            valid_rtrn_y = np.insert(valid_rtrn_y, valid_rand_int, valid_y[valid_rand_int], 0)
            valid_rtrn_y = np.delete(valid_rtrn_y, valid_rand_int+1, 0)
            valid_cnt += 1

        done = train_cnt < t and valid_cnt < v
    # compare ls and train and valid rtrn
    def assure(ls, train_x, train_y, valid_x, valid_y):
        print("\n")
        print("Printing assurance info")
        alib.dots(5)
        print("\n")
        print("Shape of ls: {}\nShape of train_x: {}\nShape of train_y: {}\nShape of valid_x: {}\n"
              "Shape of valid_y: {}".format((len(ls), len(ls[0]), len(ls[0][0])), train_x.shape, train_y.shape, valid_x.shape,
                                            valid_y.shape))

    # can put more tests for assurance progressively
    assure(ls[0]+ls[1], train_rtrn_x, train_rtrn_y, valid_rtrn_x, valid_rtrn_y)
    return [train_rtrn_x, train_rtrn_y], [valid_rtrn_x, valid_rtrn_y]

if __name__ == '__main__':
    fetch_data_single("4a", False)
