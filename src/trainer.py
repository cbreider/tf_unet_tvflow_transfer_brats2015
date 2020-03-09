"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

import tensorflow as tf
import logging
import os
import src.utils.data_utils as dutil
import src.utils.io_utils as ioutil
import numpy as np
from src.tf_convnet.caffe2tensorflow_mapping import load_pre_trained_caffe_variables
from src.utils.enum_params import Cost, Optimizer, RestoreMode, TrainingModes, Scores, ScoresLong
from src.tf_data_pipeline_wrapper import ImageData
from configuration import TrainingParams
from src.validator import Validator
import src.validator as vali
import collections
import random


class Trainer(object):

    def __init__(self, net, data_provider_train, data_provider_val, out_path, train_config, mode,  restore_path=None,
                 caffemodel_path=None, restore_mode=RestoreMode.COMPLETE_SESSION, data_provider_test=None, fold_nr=0):

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
        self._dropout_conv1 = 1.0 - self.config.dropout_rate_conv1
        self._dropout_conv2 = 1.0 - self.config.dropout_rate_conv2
        self._dropout_pool = 1.0 - self.config.dropout_rate_pool
        self._dropout_tconv = 1.0 - self.config.dropout_rate_tconv
        self._write_graph = self.config.write_graph
        self._caffemodel_path = caffemodel_path
        self._restore_path = restore_path
        self._restore_mode = restore_mode
        self._display_step = self.config.display_step
        self._log_mini_batch_stats = self.config.log_mini_batch_stats
        self._store_feature_maps = self.config.store_val_feature_maps
        self._store_result_images = self.config.store_val_images
        self._early_stopping = self.config.early_stopping
        self.data_provider_test = data_provider_test
        self._fold_nr = fold_nr

    def _get_optimizer(self, global_step):
        tvars = tf.trainable_variables()
        if self.optimizer_name == Optimizer.MOMENTUM:
            self.learning_rate = self.config.momentum_args.pop("learning_rate", 0.2)
            self.decay_rate = self.config.momentum_args.pop("decay_rate", 0.95)
            self.momentum = self.config.momentum_args.pop("momentum", 0.2)
            self.decay_steps = self.config.momentum_args.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=self.momentum,
                                                   **self.config.momentum_args).minimize(self.net.cost,
                                                                                         var_list=tvars,
                                                                                         global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAM:
            self.learning_rate = self.config.adam_args.pop("learning_rate", 0.001)
            self.decay_rate = self.config.adam_args.pop("decay_rate", 0.95)
            self.decay_steps = self.config.adam_args.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_node,
                **self.config.adam_args).minimize(self.net.cost,
                                                  var_list=tvars,
                                                  global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAGRAD:
            self.learning_rate = self.config.adagrad_args.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(self.learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(
                                                learning_rate=self.learning_rate_node,
                                                **self.config.adagrad_args).minimize(self.net.cost,
                                                                                     var_list=tvars,
                                                                                     global_step=global_step)

        else:
            raise ValueError()

        return optimizer

    def _initialize(self):

        global_step = tf.Variable(0, name="global_step", trainable=False)
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                               name="norm_gradients", trainable=False)

        if self.net.summaries and self._norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        return init

    def train(self):
        """
        Lauches the training process

        """

        init = self._initialize()

        logging.info(
            "Start optimizing model with,"
            "Optimizer: {}, "
            "Learning rate {}, Decay rate {}, Decay steps {},"
            "Nr of Epochs: {}, "
            "Epoch size: {}, "
            "Keep prob {} {} {} {}".format(self.optimizer_name, self.learning_rate, self.decay_rate, self.decay_steps,
                                  self._n_epochs, self._training_iters, self._dropout_conv1, self._dropout_conv2,
                                     self._dropout_pool, self._dropout_tconv))

        save_path = os.path.join(self.output_path, "model.ckpt")
        if self._n_epochs == 0:
            return save_path

        with tf.Session() as sess:
            if self._write_graph:
                tf.train.write_graph(sess.graph_def, self.output_path, "graph.pb", False)

            sess.run(init)
            sess.run(self.data_provider_train.init_op)

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

            # load epoch and step count from prev run if exists
            step_file = os.path.join(self.output_path, "epoch.txt")
            init_step = 0
            epoch = 0
            s_train = 0
            if os.path.isfile(step_file) and self._restore_mode != RestoreMode.ONLY_BASE_NET:
                f = open(step_file, "r")
                fl = f.readlines()
                init_step = int(fl[0])
                epoch = int(fl[1])

            pred_shape, init_loss = self.run_validtaion(sess, epoch, init_step, summary_writer_validation,
                                                        log_tf_summary=True if epoch == 0 else False, mini_validation=True)

            avg_gradients = None

            if init_step != 0 and self._restore_path is not None:
                logging.info("Resuming Training at epoch {} and total step {}". format(epoch, init_step))

            logging.info("Start optimization...")
            avg_score_vals_batch = []
            avg_score_vals_epoch = []
            last_validation_scores = [init_loss, init_loss]
            zero_counter = 0

            try:
                for step in range(init_step, self._n_epochs*self._training_iters):
                    s_train += 1
                    # renitialze dataprovider if looped through a hole dataset
                    if (s_train*self.config.batch_size_train+self.config.batch_size_train) > self.data_provider_train.size:
                        sess.run(self.data_provider_train.init_op)
                        s_train = 0

                    batch_x, batch_y, batch_tv = sess.run(self.data_provider_train.next_batch)

                    if step == 0:
                        self.output_minibatch_stats(sess, step, batch_x, dutil.crop_to_shape(batch_y, pred_shape),
                                                    summary_writer=summary_writer_training)

                    # Run optimization op (backprop)
                    _, loss, cs, err, acc, iou, dice, d_complete, d_core, d_enhancing, l1, l2, lr, gradients, pred = sess.run(
                        (self.optimizer, self.net.cost, self.net.cross_entropy, self.net.error, self.net.accuracy,
                         self.net.iou_coe, self.net.dice, self.net.dice_complete, self.net.dice_core,
                         self.net.dice_enhancing, self.net.l1regularizers, self.net.l2regularizers,
                         self.learning_rate_node, self.net.gradients_node, self.net.predicter),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: dutil.crop_to_shape(batch_y, pred_shape),
                                   self.net.keep_prob_conv1: self._dropout_conv1,
                                   self.net.keep_prob_conv2: self._dropout_conv2,
                                   self.net.keep_prob_pool: self._dropout_pool,
                                   self.net.keep_prob_tconv: self._dropout_tconv})

                    if np.max(batch_y) == 0.0:
                        zero_counter += 1

                    avg_score_vals_batch.append([loss, cs, err, acc, iou, dice, d_complete, d_core, d_enhancing, l1, l2])
                    avg_score_vals_epoch.append([loss, cs, err, acc, iou, dice, d_complete, d_core, d_enhancing])

                    #x = random.randint(1, 50)
                    #if x == 50:
                    #    Validator.store_prediction("{}_{}".format(epoch, step), self.mode, self.output_path,
                    #                               batch_x, batch_y, batch_tv, pred,
                    #                               gt_is_one_hot=False if self.net.cost == Cost.MSE else True)

                    if self.net.summaries and self._norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % self._display_step == 0 and step != 0:
                        if self._log_mini_batch_stats:
                            self.output_minibatch_stats(sess, step, batch_x, dutil.crop_to_shape(batch_y, pred_shape))

                        self.run_summary(sess, summary_writer_training, step, batch_x,
                                         dutil.crop_to_shape(batch_y, pred_shape))

                        avg_score_vals_batch = np.mean(np.array(avg_score_vals_batch), axis=0)

                        scores = collections.OrderedDict()

                        scores[Scores.LOSS] = avg_score_vals_batch[0]
                        scores[Scores.CE] = avg_score_vals_batch[1]
                        scores[Scores.ERROR] = avg_score_vals_batch[2]
                        scores[Scores.ACC] = avg_score_vals_batch[3]
                        scores[Scores.IOU] = avg_score_vals_batch[4]
                        scores[Scores.DSC] = avg_score_vals_batch[5]
                        scores[Scores.DSC_COMP] = avg_score_vals_batch[6]
                        scores[Scores.DSC_CORE] = avg_score_vals_batch[7]
                        scores[Scores.DSC_EN] = avg_score_vals_batch[8]
                        scores[Scores.L1] = avg_score_vals_batch[9]
                        scores[Scores.L2] = avg_score_vals_batch[10]

                        self.write_tf_summary_scores(step, scores, summary_writer_training)
                        self.write_log_string("Iteration {} Average:".format(step), scores)
                        avg_score_vals_batch = []

                    if step % self._training_iters == 0 and step != 0:
                        epoch += 1
                        self.output_training_epoch_stats(epoch, np.mean(np.array(avg_score_vals_epoch), axis=0), lr)
                        logging.info("EPOCH {}: Nr. of empty batches: {}".format(epoch, zero_counter))
                        zero_counter = 0
                        avg_score_vals_epoch = []
                        pred_shape, val_score = self.run_validtaion(sess, epoch, step, summary_writer_validation)
                        logging.info("Saving Session and Model ...")
                        save_path = self.net.save(sess, save_path)
                        # save epoch and step
                        self.save_step_nr(step_file, step, epoch)

                        if (last_validation_scores[0] < val_score > last_validation_scores[1]) and self._early_stopping:
                            logging.info("Stooping training because of validation convergence...")
                            break
                        last_validation_scores[0] = last_validation_scores[1]
                        last_validation_scores[1] = val_score

                logging.info("Optimization Finished!")

                if self.data_provider_test:
                    vali.run_test(sess, net=self.net, data_provider_test=self.data_provider_test, mode=self.mode,
                                  nr=self._fold_nr, out_path=self.output_path)

                return save_path

            except KeyboardInterrupt:
                logging.info("Training canceled by user.")
                save = ioutil.query_yes_no("Would you like to save tf session and model?", default=None)
                if save:
                    logging.info("Saving session model...")
                    save_path = self.net.save(sess, save_path)
                    self.save_step_nr(step_file, step, epoch)
                else:
                    logging.info("Quitting without saving...")
                logging.info("Done! Bye Bye")
                return save_path

    def save_step_nr(self, step_file, step, epoch):
        outF = open(step_file, "w")
        outF.write("{}".format(step + 1))
        outF.write("\n")
        outF.write("{}".format(epoch))
        outF.close()

    def run_validtaion(self, sess, epoch, step, summary_writer, log_tf_summary=True, mini_validation=False):
        logging.info("Running Validation for epoch {}...".format(epoch))
        epoch_out_path = os.path.join(self.output_path, "Epoch_{}".format(epoch))

        pred_shape, scores = Validator(sess, self.net, self.data_provider_val, epoch_out_path, mode=self.mode,
                                       mini_validation=mini_validation, nr=epoch,
                                       store_feature_maps=self._store_feature_maps,
                                       store_predictions=self._store_result_images).run_validation()
        if log_tf_summary:
            self.write_tf_summary_scores(step, scores, summary_writer)

        self.write_log_string("EPOCH {} Verification:".format(epoch), scores)

        return pred_shape, scores[Scores.LOSS]

    def output_training_epoch_stats(self, epoch, val_scores, lr):
        scores = collections.OrderedDict()

        scores[Scores.LOSS] = val_scores[0]
        scores[Scores.CE] = val_scores[1]
        scores[Scores.ERROR] = val_scores[2]
        scores[Scores.ACC] = val_scores[3]
        scores[Scores.IOU] = val_scores[4]
        scores[Scores.DSC] = val_scores[5]
        scores[Scores.DSC_COMP] = val_scores[6]
        scores[Scores.DSC_CORE] = val_scores[7]
        scores[Scores.DSC_EN] = val_scores[8]
        scores[Scores.LR] = lr

        self.write_log_string("Epoch {} Training Average:".format(epoch), scores)

    def output_minibatch_stats(self, sess, step, batch_x, batch_y, summary_writer=None):

        loss, ce, err, acc, iou, dice, d_complete, d_core, d_enhancing = sess.run(
            (self.net.cost, self.net.cross_entropy, self.net.error, self.net.accuracy,
             self.net.iou_coe, self.net.dice, self.net.dice_complete, self.net.dice_core,
             self.net.dice_enhancing),
            feed_dict={self.net.x: batch_x,
                       self.net.y: batch_y,
                       self.net.keep_prob_conv1: 1.0,
                       self.net.keep_prob_conv2: 1.0,
                       self.net.keep_prob_pool: 1.0,
                       self.net.keep_prob_tconv: 1.0})
        scores = collections.OrderedDict()

        scores[Scores.LOSS] = loss
        scores[Scores.CE] = ce
        scores[Scores.ERROR] = err
        scores[Scores.ACC] = acc
        scores[Scores.IOU] = iou
        scores[Scores.DSC] = dice
        scores[Scores.DSC_COMP] = d_complete
        scores[Scores.DSC_CORE] = d_core
        scores[Scores.DSC_EN] = d_enhancing

        self.write_log_string("Iter {:} Minibatch:".format(step), scores)
        if summary_writer:
            self.write_tf_summary_scores(step, scores, summary_writer=summary_writer)

    def run_summary(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str = sess.run(self.summary_op, feed_dict={self.net.x: batch_x,
                                                           self.net.y: batch_y,
                                                           self.net.keep_prob_conv1: 1.0,
                                                           self.net.keep_prob_conv2: 1.0,
                                                           self.net.keep_prob_pool: 1.0,
                                                           self.net.keep_prob_tconv: 1.0})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    @staticmethod
    def write_tf_summary_scores(step, scores, summary_writer):
        summary = tf.Summary()
        for k, v in scores.items():
            if v != -1.:
                summary.value.add(tag=ScoresLong[k], simple_value=v)

        summary_writer.add_summary(summary, step)
        summary_writer.flush()

    @staticmethod
    def write_log_string(base, scores):
        l_string = base
        for k, v in scores.items():
            if v != -1.:
                l_string = "{} {}={:.6f},".format(l_string, k.value, v)
        l_string = l_string[:-1]
        logging.info(l_string)


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients

