"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019

adapted from jakeret
source: https://github.com/jakeret/tf_unet.git
"""

import tensorflow as tf
import logging
import os
import src.utils.data_utils as util
import numpy as np
from src.tf_unet.caffe2tensorflow_mapping import load_pre_trained_caffe_variables
from src.utils.enum_params import DataModes, Cost, TrainingModes, Optimizer, RestoreMode


class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    def __init__(self, net, norm_grads=False, optimizer=Optimizer.MOMENTUM, opt_kwargs={}):
        self.net = net
        self.norm_grads = norm_grads
        self.optimizer_name = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, global_step):
        if self.optimizer_name == Optimizer.MOMENTUM:
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            decay_steps = self.opt_kwargs.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                    **self.opt_kwargs).minimize(self.net.cost,
                                                                                global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAM:
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            decay_steps = self.opt_kwargs.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                global_step=global_step,
                                                                decay_steps=decay_steps,
                                                                decay_rate=decay_rate,
                                                                staircase=True)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_node,
                **self.opt_kwargs).minimize(
                                            self.net.cost,
                                            global_step=global_step)
        elif self.optimizer_name == Optimizer.ADAGRAD:
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(
                                                learning_rate=self.learning_rate_node,
                                                **self.opt_kwargs).minimize(
                                                                            self.net.cost,
                                                                            global_step=global_step)

        else:
            raise ValueError()

        return optimizer

    def _initialize(self, training_iters, output_path):

        global_step = tf.Variable(0, name="global_step")
        self.out_path = output_path
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                                name="norm_gradients")

        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)
        tf.summary.scalar('error', self.net.error)
        tf.summary.scalar('error_rate', self.net.error_rate)
        if not self.net.cost == Cost.MSE:
            tf.summary.scalar('cross_entropy', self.net.cross_entropy)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        return init

    def train(self, data_provider_train, data_provider_val, out_path, training_iters=10, epochs=100, dropout=0.75,
              display_step=1, write_graph=False, restore_path=None, caffemodel_path=None,
              restore_mode=RestoreMode.COMPLETE_SESSION):
        """
        Lauches the training process

        :param data_provider_train: callable returning training and verification data
        :param data_provider_val: callable returning training and verification data
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore_path: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param out_path: path where to save
        :param restore_mode: selct mode to restore. Only used if restore path is set
        :param caffemodel_path: Set path to lad pretrained caffe model

        """
        logging.info(
            "Start optimizing model with,"
            "Optimizer: {}, "
            "Nr of Epochs: {}, "
            "Nr of Training Iters: {},"
            "Keep prob {}".format(self.optimizer_name, epochs, training_iters, dropout))

        init = self._initialize(training_iters=training_iters, output_path=out_path)

        save_path = os.path.join(self.out_path, "model.ckpt")
        if epochs == 0:
            return save_path

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, self.out_path, "graph.pb", False)

            sess.run(init)
            sess.run(data_provider_val.init_op)

            if caffemodel_path and restore_path:
                raise ValueError("Could not load both: Caffemodel and tf checkpoint")

            if restore_path:
                ckpt = tf.train.get_checkpoint_state(restore_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path, restore_mode=restore_mode)

            if caffemodel_path:
                load_pre_trained_caffe_variables(session=sess, file_path=caffemodel_path)

            train_summary_path = os.path.join(self.out_path, "training_summary")
            val_summary_path = os.path.join(self.out_path, "validation_summary")

            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            if not os.path.exists(val_summary_path):
                os.makedirs(val_summary_path)
            summary_writer_training = tf.summary.FileWriter(train_summary_path,
                                                            graph=sess.graph)
            summary_writer_validation = tf.summary.FileWriter(val_summary_path,
                                                            graph=sess.graph)

            test_x, test_y = sess.run(data_provider_val.next_batch)
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init", summary_writer_validation, 0, 0)
            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                sess.run(data_provider_train.init_op)

                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = sess.run(data_provider_train.next_batch)

                    # Run optimization op (backprop)
                    if step == 0:
                        self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                    util.crop_to_shape(batch_y, pred_shape))
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={self.net.x: batch_x,
                                    self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                    self.net.keep_prob: dropout})

                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % display_step == 0 and step != 0:
                        self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                    util.crop_to_shape(batch_y, pred_shape))

                    total_loss += loss

                test_x, test_y = sess.run(data_provider_val.next_batch)
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, util.crop_to_shape(test_y, pred_shape),
                                      "epoch_%s" % epoch, summary_writer_validation, step, epoch)

                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name, summary_writer, step, epoch):
        loss, acc, err, prediction = self.run_summary(sess, summary_writer, step, batch_x, batch_y)

        pred_shape = prediction.shape

        logging.info("EPOCH {}: Verification error= {:.1f}%, loss= {:.6f}".format(epoch, err, loss))

        img = util.combine_img_prediction(batch_x, batch_y, prediction,
                                          mode=1 if self.net.cost_function == Cost.MSE else 0)
        util.save_image(img, "%s/%s.jpg" % (self.out_path, name))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.9f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):

        loss, acc, err, predictions = self.run_summary(sess, summary_writer, step, batch_x, batch_y)
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            err))

    def run_summary(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, err, predictions = sess.run([self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.error_rate,
                                                        self.net.predicter],
                                                        feed_dict={self.net.x: batch_x,
                                                                    self.net.y: batch_y,
                                                                    self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        return loss, acc, err, predictions


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients

