from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_BraTSData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "data/BraTS2018/MICCAI_BraTS_2018_Data_Training/HGG", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize/ evalutate")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224
NUM_INPUT_CHANNEL = 4

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            if (name == 'conv1_1'):
                kernel_shape = kernels.shape[:2] + (4, ) + kernels.shape[3:]
            else:
                kernel_shape = kernels.shape
            
            bias_shape = bias.shape
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            new_kernel = np.zeros(kernel_shape)
            new_kernel_shape = np.transpose(new_kernel, (1, 0, 2, 3)).shape
            print(f"new kernel shape: {new_kernel_shape}")
            new_bias = np.zeros(bias_shape)
            new_bias_shape = new_bias.reshape(-1).shape[0]
            print(f"new bias shape: {new_bias_shape}")

            kernels = utils.weight_variable(shape=new_kernel_shape, name=name + "_w" )
            bias = utils.bias_variable(shape=[new_bias_shape], name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
        print(f"VGG-19 {name} layer: {current.shape}")
    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    # get pre-trained model
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    # mean = model_data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    # processed_image = utils.process_image(image, mean_pixel)
    processed_image = image

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        print(f"image net loaded")

        ####################### SHAPE NETWORK ##############################
        shape_net_input_layer = image_net["relu1_2"]
        print(f"shape net input layer: {shape_net_input_layer.shape}")

        W_s1_1 = utils.weight_variable([3, 3, 64, 128], name="W_s1_1")
        b_s1_1 = utils.bias_variable([128], name='b_s1_1')
        conv_s1_1 = utils.conv2d_basic(shape_net_input_layer, W_s1_1, b_s1_1)
        relu_s1_1 = tf.nn.relu(conv_s1_1, name="relu_s1_1")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s1_1)
        relu_dropout_s1_1 = tf.nn.dropout(relu_s1_1, keep_prob=keep_prob)

        W_s1_2 = utils.weight_variable([3, 3, 128, 128], name="W_s1_2")
        b_s1_2 = utils.bias_variable([128], name="b_s1_2")
        conv_s1_2 = utils.conv2d_basic(relu_dropout_s1_1, W_s1_2, b_s1_2)
        relu_s1_2 = tf.nn.relu(conv_s1_2, name="relu_s1_2")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s1_2)
        relu_dropout_s1_2 = tf.nn.dropout(relu_s1_2, keep_prob=keep_prob)

        W_s2_1 = utils.weight_variable([3, 3, 128, 256], name="W_s2_1")
        b_s2_1 = utils.bias_variable([256], name="b_s2_1")
        conv_s2_1 = utils.conv2d_basic(relu_dropout_s1_2, W_s2_1, b_s2_1)
        relu_s2_1 = tf.nn.relu(conv_s2_1, name="relu_s2_1")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s2_1)
        relu_dropout_s2_1 = tf.nn.dropout(relu_s2_1, keep_prob=keep_prob)

        W_s2_2 = utils.weight_variable([3, 3, 256, 256], name="W_s2_2")
        b_s2_2 = utils.bias_variable([256], name="b_s2_2")
        conv_s2_2 = utils.conv2d_basic(relu_dropout_s2_1, W_s2_2, b_s2_2)
        relu_s2_2 = tf.nn.relu(conv_s2_2, name="relu_s2_2")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s2_2)
        relu_dropout_s2_2 = tf.nn.dropout(relu_s2_2, keep_prob=keep_prob)

        W_s2_3 = utils.weight_variable([3, 3, 256, 256], name="W_s2_3")
        b_s2_3 = utils.bias_variable([256], name="b_s2_3")
        conv_s2_3 = utils.conv2d_basic(relu_dropout_s2_2, W_s2_3, b_s2_3)
        relu_s2_3 = tf.nn.relu(conv_s2_3, name="relu_s2_3")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s2_3)
        relu_dropout_s2_3 = tf.nn.dropout(relu_s2_3, keep_prob=keep_prob)

        W_s2_4 = utils.weight_variable([3, 3, 256, 256], name="W_s2_4")
        b_s2_4 = utils.bias_variable([256], name="b_s2_4")
        conv_s2_4 = utils.conv2d_basic(relu_dropout_s2_3, W_s2_4, b_s2_4)
        relu_s2_4 = tf.nn.relu(conv_s2_1, name="relu_s2_4")
        if FLAGS.debug:
            utils.add_activation_summary(relu_s2_4)
        relu_dropout_s2_4 = tf.nn.dropout(relu_s2_4, keep_prob=keep_prob)
        print(f"Shape Network last layer: {relu_dropout_s2_4.shape}")
        
        ####################################################################

        conv_final_layer = image_net["conv5_3"]
        pool5 = utils.max_pool_2x2(conv_final_layer)
        print(f"VGG-19 pool5 layer: {pool5.shape}")

        # [height, width, in_channel, out_channel]
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6") 
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        print(f"VGG-19 conv6 layer: {conv6.shape}")
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        print(f"VGG-19 conv7 layer: {conv7.shape}")
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        print(f"VGG-19 conv8 layer: {conv8.shape}")
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # stride is 2(factored by 2) for this transpose upsampling
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        print(f"FCN conv_t1 layer: {conv_t1.shape}")
        # combines 2x upsampled layer and pool4 layer
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        print(f"FCN fuse_1 layer: {fuse_1.shape}")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # deconvolution fuse1 with W_t2
        # stride is 2(factored by 2) for this transpose upsampling
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        print(f"FCN conv_t2 layer: {conv_t2.shape}")
        # combines 2x upsampled previous fused layer and pool3 layer
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        print(f"FCN fuse_2 layer: {fuse_2.shape}")

        shape = tf.shape(image)
        # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        # W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        # b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 256])
        W_t3 = utils.weight_variable([16, 16, 256, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([256], name="b_t3")
        # finally upsample the layer to replace input size
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        print(f"FCN conv_t3 layer: {conv_t3.shape}")

        ######################### concat shape network and FCN ############################
        concat = tf.concat([relu_dropout_s2_4, conv_t3], axis=-1, name="concat")
        
        W_label = utils.weight_variable([1, 1, 512, NUM_OF_CLASSESS], name="W_label")
        b_label = utils.bias_variable([NUM_OF_CLASSESS], name="b_label")
        conv_label = utils.conv2d_basic(concat, W_label, b_label)

        #################################################################################3

        # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        annotation_pred = tf.argmax(conv_label, dimension=3, name="prediction")
        
    # return tf.expand_dims(annotation_pred, dim=3), conv_t3
    return tf.expand_dims(annotation_pred, dim=3), conv_label


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    # image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # 4 for t1, t1ce, t2 and flair
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 4], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        print("reading training dataset... wait a moment...")
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                 # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image((valid_annotations[itr] * 100).astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image((pred[itr] * 100).astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "evaluate":
        gt_tumor = 0
        seg_tumor = 0
        overlapped = 0
        for i in range(len(valid_records)):
            valid_images, valid_annotations = validation_dataset_reader.get_single_brain(i)
            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)

            for itr in range(valid_images.shape[0]):
                gt = np.asarray(valid_annotations[itr]).astype(np.bool)
                seg = np.asarray(pred[itr]).astype(np.bool)
                pixels = len(gt) * len(gt)
                
                gt_tumor += gt.sum()
                seg_tumor += seg.sum()
                overlapped += np.logical_and(gt, seg).sum()
        dice = 2 * overlapped / (gt_tumor + seg_tumor)
        print(f"DICE COEFFICIENT: {dice}")

if __name__ == "__main__":
    tf.app.run()
