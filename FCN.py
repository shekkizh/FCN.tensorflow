from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import os
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("test_data_dir", "test_results/", "path to test dataset")
tf.flags.DEFINE_float("initial_learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224
fine_tuning = True


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
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        annotation_pred_width_one_channel = tf.expand_dims(annotation_pred, dim=3)

    return annotation_pred_width_one_channel, conv_t3


def train(loss_val, var_list):
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step=global_step,
                                               decay_steps=5e4, decay_rate=0.1)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    return optimizer.minimize(loss_val)

    # grads = optimizer.compute_gradients(loss_val, var_list=var_list) # compute gradients by the loss_val
    # if FLAGS.debug:
    #     # print(len(var_list))
    #     for grad, var in grads:
    #         utils.add_gradient_summary(grad, var)
    # return optimizer.apply_gradients(grads) # use the gradients results to flush the varibles


def cross_entroy_loss(y_true, y_pred, num_classes=NUM_OF_CLASSESS):
    '''
    calculate the loss from the y_true and the y_pred.
    :param y_true: tensor,float -[batch-size,width,heitht,num_classes].
        use vgg_fcn.upscore as y_true.
    :param y_pred: y_pred tensor,int32 -[batch_size, width,height, num_classes]
        the ground truth of data
    :param num_classes:
    :return: Loss tensor of type float
    '''
    with tf.name_scope("loss"):
        y_pred = tf.reshape(y_pred, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

        softmax = tf.nn.softmax(y_pred) + epsilon

        corss_entropy = -tf.reduce_sum(y_true * tf.log(softmax), reduction_indices=[1])

        corss_entropy_mean = tf.reduce_mean(corss_entropy, name="xentropy_mean")
        tf.add_to_collection("losses", corss_entropy_mean)
        loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return loss


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))
    # logits shape is (?,height,width,151)
    annotation_tmp = tf.squeeze(annotation, axis=3)
    annotation_channels = tf.one_hot(indices=annotation_tmp, depth=NUM_OF_CLASSESS, axis=3)

    loss = cross_entroy_loss(annotation_channels, logits, num_classes=NUM_OF_CLASSESS)
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    # train_op = train(loss, trainable_var)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step=global_step,
                                               decay_steps=5e4, decay_rate=0.1)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # global iter times
    add_global = global_step.assign_add(1)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    with tf.Session() as sess:
        print("Setting up Saver...")
        saver = tf.train.Saver()

        # create two summary writers to show training loss and validation loss in the same graph
        # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
        train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

        sess.run(tf.global_variables_initializer())
        if fine_tuning:
            ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")

        if FLAGS.mode == "train":
            for itr in xrange(MAX_ITERATION):
                train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

                sess.run([train_op, add_global], feed_dict=feed_dict)

                if itr % 10 == 0:
                    train_loss, summary_str, lr_rate = sess.run([loss, loss_summary, learning_rate], feed_dict=feed_dict)
                    train_writer.add_summary(summary_str, itr)
                    valid_loss, summary_sva = sess.run([loss, loss_summary],
                                                       feed_dict={image: valid_images, annotation: valid_annotations,
                                                                  keep_probability: 1.0})
                    # add validation loss to TensorBoard
                    validation_writer.add_summary(summary_sva, itr)
                    print("Step: %d,learning_rate:%g , Train_loss:%g---------> "
                          "Validation_loss: %g" % (itr, lr_rate, train_loss, valid_loss))
                if itr % 500 == 0:
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=itr)

        elif FLAGS.mode == "visualize":
            valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                        keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)
            if not os.path.exists(FLAGS.test_data_dir):
                os.makedirs(FLAGS.test_data_dir)
            for itr in range(FLAGS.batch_size):
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.test_data_dir, name="inp_" + str(5 + itr))
                utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.test_data_dir,
                                 name="gt_" + str(5 + itr))
                utils.save_image(pred[itr].astype(np.uint8), FLAGS.test_data_dir, name="pred_" + str(5 + itr))
                print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
