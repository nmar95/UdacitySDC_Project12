import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.save_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer_3, layer_4, layer_7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Implement function
    # FCN Decoder Classroom mentions alot of this
    # Will needto use skip connections and up sampling how-to-prepare-augment-images-for-neural-network
    # This is the meat fo the project. This is where you are creating the architecture.
    # must do a kernal regularizer for every layer.

    # The paper in the decoder lesson also mentions the architecture
    # Upsample by 2, then by 2, and lastly by 8

    # 1x1 conv of layer 7
    layer7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Upsample layer 4
    layer4_upsample = tf.layers.conv2d_transpose(layer7_conv, num_classes, 4, strides= (2, 2), padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    #1x1 conv of layer 4
    layer4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Skip connection for layer 4
    layer4_skip = tf.add(layer4_upsample, layer4_conv)

    # Upsample layer 3
    layer3_upsample = tf.layers.conv2d_transpose(layer4_skip, num_classes, 4, strides= (2, 2), padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 conv of layer 3
    layer3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Skip connection for layer 3
    layer3_skip = tf.add(layer3_upsample, layer3_conv)

    # Final Upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3_skip, num_classes, 16, strides= (8, 8), padding= 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Output the last layer of the tensor
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Implement function
    # In classroom -> Scene Understanding -> Classification and Loss
    # Recomended to use an adam optimizer

    """
    From Classroom:
    The final step is to define a loss.
    That way, we can approach training a FCN just like we would
    approach training a normal classification CNN.

    In the case of a FCN, the goal is to assign each pixel to the appropriate class.
    We already happen to know a great loss function for this setup, cross entropy loss!
    Remember the output tensor is 4D so we have to reshape it to 2D:
    """
    # From classroom - Logits is now a 2D tensor where each row represents a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # From here we can just use standard cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    # Train using an Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Minimize loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Implement function
    # This all comes down to understanding what the get_batches_fn is helping you
    keep_probability = 0.8
    learning_rate_value = 0.001

    for epochs in range(epochs):
        for image, label in get_batches_fn(batch_size):
            # Training
            _, loss = sess.run([train_op, cross_entropy_loss],
            feed_dict={input_image: image, correct_label: label, keep_prob: keep_probability, learning_rate:learning_rate_value})

        print("Epoch {}/{}...".format(epoch_i+1, epochs), "Training Loss: {:.4f}...".format(loss))
tests.test_train_nn(train_nn)

#Run function will actually run the model
def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    #Parameters
    epochs = 15
    batch_size = 16
    learning_rate = tf.constant(0.0001)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))

        # Output of this is the final layer of the NN to build the FCN
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)



if __name__ == '__main__':
    run()
