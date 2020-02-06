"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019
"""

import tensorflow as tf
import logging
import os
import src.utils.data_utils as dutil
import src.utils.io_utils as ioutil
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
        self._log_mini_batch_stats = self.config.log_mini_batch_stats

    def _get_optimizer(self, global_step):
        if self.optimizer_name == Optimizer.MOMENTUM:
            self.learning_rate = self.config.momentum_args.pop("learning_rate", 0.2)
            self.decay_rate = self.config.momentum_args.pop("decay_rate", 0.95)
            self.momentum = self.config.momentum_args.pop("momentum", 0.2)
            self.decay_steps = self.config.momentum_args.pop("decay_steps", 10000)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                 global_step=self.global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=self.momentum,
                                                   **self.config.momentum_args).minimize(self.net.cost,
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
                                                  global_step=global_step)

        elif self.optimizer_name == Optimizer.ADAGRAD:
            self.learning_rate = self.config.adagrad_args.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(self.learning_rate, name="learning_rate")

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
            "Epoch size: {},"
            "Keep prob {}".format(self.optimizer_name, self.learning_rate, self.decay_rate, self.decay_steps,
                                  self._n_epochs, self._training_iters, self._dropout))

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

            pred_shape = self.run_validation(epoch, sess, init_step, summary_writer_validation, save_path, mini=True,
                                             log=False if epoch != 0 else True)

            avg_gradients = None

            if init_step != 0 and self._restore_path is not None:
                logging.info("Resuming Training at epoch {} and total step {}". format(epoch, init_step))

            epoch_size = int(self.data_provider_train.size / self.config.batch_size_train)

            logging.info("Start optimization...")
            avg_score_vals_batch = []
            avg_score_vals_epoch = []
            try:
                for step in range(init_step, self._n_epochs*self._training_iters):
                    s_train += 1
                    # renitialze dataprovider if looped through a hole dataset
                    if (s_train*self.config.batch_size_train+self.config.batch_size_train) > self.data_provider_train.size:
                        sess.run(self.data_provider_train.init_op)
                        s_train = 0

                    batch_x, batch_y, batch_tv = sess.run(self.data_provider_train.next_batch)

                    # Run optimization op (backprop)
                    if step == 0:
                        self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                    dutil.crop_to_shape(batch_y, pred_shape), write=True,
                                                    log_mini_batch_stats=True)
                    _, loss, cs, dice, err, err_r, acc, lr, gradients, pred, iou = sess.run(
                        (self.optimizer, self.net.cost, self.net.cross_entropy, self.net.dice, self.net.error,
                         self.net.error_rate, self.net.accuracy, self.learning_rate_node, self.net.gradients_node,
                         self.net.predicter, self.net.iou_coe),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: dutil.crop_to_shape(batch_y, pred_shape),
                                   self.net.keep_prob: self._dropout})
                    avg_score_vals_batch.append([loss, cs, dice, err, err_r, acc, iou])
                    avg_score_vals_epoch.append([loss, cs, dice, err_r, acc, iou])

                    #self.store_prediction("{}_{}".format(epoch, step), self.output_path,
                    #                      batch_x, batch_x, batch_tv, pred)

                    if self.net.summaries and self._norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % self._display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer_training, step, batch_x,
                                                    dutil.crop_to_shape(batch_y, pred_shape), write=True,
                                                    log_mini_batch_stats=self._log_mini_batch_stats)
                        avg_score_vals_batch = np.mean(np.array(avg_score_vals_batch), axis=0)
                        self.write_tf_summary(step, avg_score_vals_batch, summary_writer_training)

                        if step != 0:
                            logging.info(
                                "Iter {:} Average: Loss= {:.6f}, Cross entropy = {:.4f}, Dice= {:.4f}, Error= {:.2f}%, "
                                "Accuracy= {:.4f}, IoU= {:.4f}".format(step, avg_score_vals_batch[0],
                                                                      avg_score_vals_batch[1], avg_score_vals_batch[2],
                                                                      avg_score_vals_batch[4], avg_score_vals_batch[5],
                                                                      avg_score_vals_batch[6]))
                            # save epoch and step
                            outF = open(step_file, "w")
                            outF.write("{}".format(step+1))
                            outF.write("\n")
                            outF.write("{}".format(epoch))
                            outF.close()
                        avg_score_vals_batch = []

                    if step % self._training_iters == 0 and step != 0:
                        epoch += 1
                        self.output_epoch_stats(epoch, np.mean(np.array(avg_score_vals_epoch)), self._training_iters, lr)
                        avg_score_vals_epoch = []
                        self.run_validation(epoch, sess, step, summary_writer_validation, save_path)

                logging.info("Optimization Finished!")

                return save_path

            except KeyboardInterrupt:
                logging.info("Training canceled by user.")
                save = ioutil.query_yes_no("Would you like to save tf session and model?", default=None)
                if save:
                    logging.info("Saving session model...")
                    save_path = self.net.save(sess, save_path)
                else:
                    logging.info("Quitting without saving...")
                logging.info("Done! Bye Bye")
                return save_path

            except Exception as e:
                logging.error(str(e))
                return None

    def run_validation(self, epoch, sess, step, summary_writer, model_save_path, mini=False, log=True):
        mini_size = 5
        vals = []
        dice_per_volume = []
        data = [[], [], [],  [], []] # x, y, tv, pred, feature maps
        shape = []
        out_p = os.path.join(self.output_path, "Epoch_{}".format(epoch))
        itr = 0
        if not os.path.exists(out_p):
            os.makedirs(out_p)
        sess.run(self.data_provider_val.init_op)
        logging.info("Running Validation for epoch {}...".format(epoch))

        set_size = int(self.data_provider_val.size / self.config.batch_size_val)
        ioutil.progress(0, set_size if not mini else (mini_size * 155))

        for i in range(int(self.data_provider_val.size / self.config.batch_size_val)):
            test_x, test_y, test_tv = sess.run(self.data_provider_val.next_batch)
            __, loss, acc, err, err_r, prediction, dice, ce, iou, feature = sess.run(
                [self.summary_op, self.net.cost,
                 self.net.accuracy, self.net.error, self.net.error_rate,
                 self.net.predicter, self.net.dice,
                 self.net.cross_entropy, self.net.iou_coe, self.net.last_feature_map],
                feed_dict={self.net.x: test_x,
                           self.net.y: test_y,
                           self.net.keep_prob: 1.})

            vals.append([loss, ce, dice, err, err_r, acc, iou])
            data[0].append(test_x)
            data[1].append(test_y)
            data[2].append(test_tv)
            data[3].append(prediction)
            data[4].append(feature)
            shape = prediction.shape

            if len(data[1]) == 155:
                dice_per_volume.append(dutil.get_hard_dice_score(np.array(data[1]), np.array(data[:][3])))
                self.store_prediction("{}_{}".format(epoch, itr), out_p,
                                      np.squeeze(np.array(data[0]), axis=1), np.squeeze(np.array(data[1]), axis=1),
                                      np.squeeze(np.array(data[2]), axis=1), np.squeeze(np.array(data[3]), axis=1))

                # safe one feature_map
                size = [8, 8]
                a = np.array(data[4][75])
                fmap = dutil.revert_zero_centering(np.squeeze(np.array(data[4][75]), axis=0))
                map_s = [fmap.shape[0], fmap.shape[1]]
                im = fmap.reshape(map_s[0], map_s[0], size[0], size[1]
                                  ).transpose(2, 0, 3, 1
                                              ).reshape(size[0] * map_s[0], size[1] * map_s[1])
                # histogram normalization
                #im = util.image_histogram_equalization(im)[0]
                ioutil.save_image(im, os.path.join(out_p, "{}_{}_fmap.jpg".format(epoch, itr)))

                data = [[], [], [],  [], []]
                itr += 1
                if mini and itr == mini_size:
                    break

            ioutil.progress(i, set_size if not mini else (mini_size * 155))

        if len(data[1]) > 0:
            self.store_prediction("{}_{}".format(epoch, itr), out_p,
                              np.squeeze(np.array(data[0]), axis=1), np.squeeze(np.array(data[1]), axis=1),
                              np.squeeze(np.array(data[2]), axis=1), np.squeeze(np.array(data[3]), axis=1))
        val_scores = np.mean(np.array(vals), axis=0)
        dp = np.mean(np.array(dice_per_volume))
        if log:
            self.write_tf_summary(step, val_scores, summary_writer, cost_val=["dice_per_volume", dp])
        logging.info(
            "EPOCH {} Per Slice: Verification loss= {:.6f}, cross entropy= {:.4f}, Dice per slice= {:.4f}, "
            "Dice per volume= {:.4f}, error= {:.2f}%, Accuracy {:.4f}, IoU= {:.4f}".format(
                epoch, val_scores[0], val_scores[1], val_scores[2], dp,
                val_scores[4], val_scores[5], val_scores[6]))
        logging.info("Saving Session and Model ...")
        save_path = self.net.save(sess, model_save_path)
        return shape

    def write_tf_summary(self, step, vals, summary_writer, cost_val=None):
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=vals[0])
        summary.value.add(tag='accuracy', simple_value=vals[5])
        summary.value.add(tag='error', simple_value=vals[4])
        summary.value.add(tag='error_rate', simple_value=vals[3])
        if not self.net.cost == Cost.MSE:
            summary.value.add(tag='cross_entropy', simple_value=vals[1])
            summary.value.add(tag='dice', simple_value=vals[2])
        if cost_val is not None:
            summary.value.add(tag=cost_val[0], simple_value=cost_val[1])

        summary_writer.add_summary(summary, step)
        summary_writer.flush()

    def store_prediction(self, name, path, batch_x, batch_y, batch_tv, prediction):
        if self.mode == TrainingModes.TVFLOW_SEGMENTATION:
            img = dutil.combine_img_prediction_tvclustering(data=batch_x, gt=batch_y, tv=batch_tv, pred=prediction)
        else:
            img = dutil.combine_img_prediction(batch_x, batch_y, prediction,
                                               mode=1 if self.net.cost_function == Cost.MSE else 0)
        ioutil.save_image(img, "%s/%s.jpg" % (path, name))

    def output_epoch_stats(self, epoch, val_scores, lr):
        logging.info(
            "EPOCH {} Average: Loss= {:.6f}, Cross entropy= {:.4f}, Dice= {:.4f}, "
            "Error= {:.2f}%, Accuracy= {:.4f}, IoU= {:.4f}, Learning rate= {:.9f}".format(
                epoch, val_scores[0], val_scores[1], val_scores[2], val_scores[3], val_scores[4], val_scores[5], lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, write=False,
                               log_mini_batch_stats=False):

        loss, acc, err, predictions, dice, ce = self.run_summary(sess, summary_writer,
                                                                 step, batch_x, batch_y, write=write)
        if log_mini_batch_stats:
            logging.info(
                "Iter {:}, Minibatch Loss= {:.6f}, Cross entropy = {:.4f}, Dice= {:.4f}, "
                "Error= {:.2f}% Accuracy= {:.4f}".format(step, loss, ce, dice, err, acc))

    def run_summary(self, sess, summary_writer, step, batch_x, batch_y, write=True):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, err, predictions, dice, ce = sess.run([self.summary_op, self.net.cost,
                                                                       self.net.accuracy, self.net.error_rate,
                                                                       self.net.predicter, self.net.dice,
                                                                       self.net.cross_entropy],
                                                                      feed_dict={self.net.x: batch_x,
                                                                                 self.net.y: batch_y,
                                                                                 self.net.keep_prob: 1.})
        if write:
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        return loss, acc, err, predictions, dice, ce


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients

