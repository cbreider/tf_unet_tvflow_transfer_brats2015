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
from src.tf_convnet.caffe2tensorflow_mapping import load_pre_trained_caffe_variables
from src.utils.enum_params import Cost, Optimizer, RestoreMode, TrainingModes
from src.tf_data_pipeline_wrapper import  ImageData
from configuration import TrainingParams


class Trainer(object):

    def __init__(self, net, data_provider_train, data_provider_val, out_path, train_config, mode,  restore_path=None,
                 caffemodel_path=None, restore_mode=RestoreMode.COMPLETE_SESSION):

        """
        Trains a convnet instance

        :param net: the unet instance to train
        :param data_provider_train: callable returning training and verification data
        :param data_provider_val: callable returning training and verification data
        :param out_path: path to store summaries and checkpoints
        :param train_config: parameters for  training # type: TrainingParams
        :param restore_path: (optional) checkpoint path to restore a tf checkpoint
        :param caffemodel_path: (optional) path to a unet caffemodel if you want to restore one
        :param restore_mode: (optional) mode how to restore a tf checkpoint
        """

        self.net = net
        self.mode = mode  # type: TrainingModes
        self.config = train_config  # type: TrainingParams
        self.data_provider_train = data_provider_train # type: ImageData
        self.data_provider_val = data_provider_val  # type: ImageData
        self.output_path = out_path
        self._norm_grads = self.config.norm_grads
        self.optimizer_name = self.config.optimizer
        self._n_epochs = self.config.num_epochs
        self._training_iters = self.config.training_iters
        self._dropout = self.config.keep_prob_dopout
        self._write_graph = self.config.write_graph
        self._caffemodel_path = caffemodel_path
        self._restore_path = restore_path
        self._restore_mode = restore_mode
        self._display_step = self.config.display_step

    def _get_optimizer(self, global_step):
        if self.optimizer_name == Optimizer.MOMENTUM:
            learning_rate = self.config.momentum_args.pop("learning_rate", 0.2)
            decay_rate = self.config.momentum_args.pop("decay_rate", 0.95)
            momentum = self.config.momentum_args.pop("momentum", 0.2)
            decay_steps = self.config.momentum_args.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.config.momentum_args).minimize(self.net.cost,
                                                                                         global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAM:
            learning_rate = self.config.adam_args.pop("learning_rate", 0.001)
            decay_rate = self.config.adam_args.pop("decay_rate", 0.95)
            decay_steps = self.config.adam_args.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_node,
                **self.config.adam_args).minimize(self.net.cost,
                                                  global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAGRAD:
            learning_rate = self.config.adagrad_args.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(
                                                learning_rate=self.learning_rate_node,
                                                **self.config.adagrad_args).minimize(self.net.cost,
                                                                                     global_step=global_step)

        else:
            raise ValueError()

        return optimizer

    def _initialize(self):

        global_step = tf.Variable(0, name="global_step")
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                               name="norm_gradients")

        if self.net.summaries and self._norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)
        tf.summary.scalar('error', self.net.error)
        tf.summary.scalar('error_rate', self.net.error_rate)
        if not self.net.cost == Cost.MSE:
            tf.summary.scalar('cross_entropy', self.net.cross_entropy)
            tf.summary.scalar('dice', self.net.dice)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        return init

    def train(self):
        """
        Lauches the training process

        """

        logging.info(
            "Start optimizing model with,"
            "Optimizer: {}, "
            "Nr of Epochs: {}, "
            "Nr of Training Iters: {},"
            "Keep prob {}".format(self.optimizer_name, self._n_epochs, self._training_iters, self._dropout))

        init = self._initialize()

        save_path = os.path.join(self.output_path, "model.ckpt")
        if self._n_epochs == 0:
            return save_path

        with tf.Session() as sess:
            if self._write_graph:
                tf.train.write_graph(sess.graph_def, self.output_path, "graph.pb", False)

            sess.run(init)
            sess.run(self.data_provider_val.init_op)

            if self._caffemodel_path and self._restore_path:
                raise ValueError("Could not load both: Caffemodel and tf checkpoint")

            if self._restore_path:
                ckpt = tf.train.get_checkpoint_state(self._restore_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path, restore_mode=self._restore_mode)

            if self._caffemodel_path:
                load_pre_trained_caffe_variables(session=sess, file_path=self._caffemodel_path)

            train_summary_path = os.path.join(self.output_path, "training_summary")
            val_summary_path = os.path.join(self.output_path, "validation_summary")

            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            if not os.path.exists(val_summary_path):
                os.makedirs(val_summary_path)
            summary_writer_training = tf.summary.FileWriter(train_summary_path,
                                                            graph=sess.graph)
            summary_writer_validation = tf.summary.FileWriter(val_summary_path,
                                                              graph=sess.graph)

            test_x, test_y, test_tv = sess.run(self.data_provider_val.next_batch)
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init", summary_writer_validation, 0, 0, test_tv)
            logging.info("Start optimization")

            avg_gradients = None

            # load epoch and step count from prev run if exists
            step_file = os.path.join(self.output_path, "epoch.txt")
            init_step = 0
            epoch = 0
            total_loss = 0
            if os.path.isfile(step_file):
                f = open(step_file, "w")
                fl = f.readlines()
                init_step = fl[0]
                epoch = fl[1]
            if init_step != 0 and self._restore_path is not None:
                logging.info("Resuming Training at epoch {} and total step {}". format(epoch, init_step))

            for step in range(init_step, self._n_epochs*self._training_iters):

                # reinitialze dataprovider if looped through a hole epoch
                sess.run(self.data_provider_train.init_op)
                # renitialze dataprovider if looped through a hole dataset
                if (step * self.config.batch_size_train) % self.data_provider_train.size == 0:
                    sess.run(self.data_provider_train.init_op)

                batch_x, batch_y, __ = sess.run(self.data_provider_train.next_batch)

                # Run optimization op (backprop)
                if step == 0:
                    self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                util.crop_to_shape(batch_y, pred_shape))
                _, loss, lr, gradients = sess.run(
                    (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                    feed_dict={self.net.x: batch_x,
                               self.net.y: util.crop_to_shape(batch_y, pred_shape),
                               self.net.keep_prob: self._dropout})

                if self.net.summaries and self._norm_grads:
                    avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.norm_gradients_node.assign(norm_gradients).eval()

                total_loss += loss

                if step % self._display_step == 0 and step != 0:
                    self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                util.crop_to_shape(batch_y, pred_shape))

                if step % self._training_iters == 0 and step != 0:
                    test_x, test_y, test_tv = sess.run(self.data_provider_val.next_batch)
                    self.output_epoch_stats(epoch, total_loss, self._training_iters, lr)
                    self.store_prediction(sess, test_x, util.crop_to_shape(test_y, pred_shape),
                                      "epoch_%s" % epoch, summary_writer_validation, step, epoch, test_tv)
                    total_loss = 0
                    save_path = self.net.save(sess, save_path)
                    # save epoch and step
                    outF = open(step_file, "w")
                    outF.write("{}".format(step+1))
                    outF.write("\n")
                    outF.write("{}".format(epoch+1))
                    outF.close()
                    epoch += 1
                    if (step * self.config.batch_size_val) % self.data_provider_val.size == 0:
                        sess.run(self.data_provider_val.init_op)

            logging.info("Optimization Finished!")

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name, summary_writer, step, epoch, batch_tv):
        loss, acc, err, prediction, dice, ce = self.run_summary(sess, summary_writer, step, batch_x, batch_y)

        pred_shape = prediction.shape

        logging.info("EPOCH {}: Verification error= {:.1f}%, loss= {:.6f}, Dice= {:.4f}, cross entropy = {:.4f}".format(
            epoch, err, loss, dice, ce))
        if self.mode == TrainingModes.TVFLOW_SEGMENTATION:
            img = util.combine_img_prediction_tvclustering(data=batch_x, gt=batch_y, tv=batch_tv, pred=prediction)
        else:
            img = util.combine_img_prediction(batch_x, batch_y, prediction,
                                              mode=1 if self.net.cost_function == Cost.MSE else 0)
        util.save_image(img, "%s/%s.jpg" % (self.output_path, name))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.9f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):

        loss, acc, err, predictions, dice, ce = self.run_summary(sess, summary_writer, step, batch_x, batch_y)
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, "
            "Minibatch error= {:.1f}%, Dice= {:.4f}, cross entropy = {:.4f}".format(step, loss, acc, err, dice, ce))

    def run_summary(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, err, predictions, dice, ce = sess.run([self.summary_op, self.net.cost,
                                                                       self.net.accuracy, self.net.error_rate,
                                                                       self.net.predicter, self.net.dice,
                                                                       self.net.cross_entropy],
                                                                      feed_dict={self.net.x: batch_x,
                                                                                 self.net.y: batch_y,
                                                                                 self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        return loss, acc, err, predictions, dice, ce


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients

