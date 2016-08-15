from __future__ import division
import tensorflow as tf
import numpy as np
import os
import tf_helpers
import tfinput
import threading
import time
from datetime import datetime
import math
from coin import Coin

class TFModel(object):
    def __init__(self, encoding, save_dir, batch_size = 100, num_epochs_per_decay = 25, moving_average_decay = .9999):
        self.encoding = encoding
        self.batch_size = batch_size
        self.num_epochs_per_decay = num_epochs_per_decay
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.moving_average_decay = moving_average_decay

    def over_fit_test(self, coinlabel, total_epochs = 5, grade = True, use_logit = False,load_save = False, balance_classes = False, wd = .0004):
         with tf.Graph().as_default():
           #weight the classes for inbalance puproses
           global_step = tf.Variable(0, trainable=False)
           feature_batch, grade_batch, name_batch = tfinput.input(coinlabel.get_overfit_test_list(), self.batch_size)
           logits = self.encoding(feature_batch, coinlabel.n_labels, do = False, weight_decay = wd)
           if use_logit:
               logit_pred, logit_cost = self._add_logit(grade_batch,name_batch, coinlabel.n_names, coinlabel.n_grades)
               logits =   tf.mul(logits, logit_pred)
           # Calculate loss.
           if grade:
               loss = tf_helpers.loss(logits, grade_batch)
           else:
               loss = tf_helpers.loss(logits, name_batch)
           if use_logit:
               loss = loss + logit_cost
           # Build a Graph that trains the model with one batch of examples and
           # updates the model parameters.
           train_op = self._train(loss, global_step)
           # Create a saver.
           saver = tf.train.Saver(tf.all_variables())
           # Build the summary operation based on the TF collection of Summaries.
           summary_op = tf.merge_all_summaries()
           # Build an initialization operation to run below.
           init = tf.initialize_all_variables()
           # Start running operations on the Graph.
           sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
           sess.run(init)
           summary_writer = tf.train.SummaryWriter(self.save_dir, sess.graph)
           ### pool the queueing processed
           sess.run(init)
           if load_save:
               ckpt = tf.train.get_checkpoint_state(self.save_dir)
               if ckpt and ckpt.model_checkpoint_path:
                   # Restores from checkpoint
                   saver.restore(sess, ckpt.model_checkpoint_path)
                   # Assuming model_checkpoint_path looks something like:
                   #   /my-favorite-path/cifar10_train/model.ckpt-0,
                   # extract global_step from it.
                   global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                   start = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
           else:
               start = 0
           tf.train.start_queue_runners(sess=sess)
           steps_per_epoch = int(len(coinlabel.get_file_list(False))/self.batch_size)
           training_iter = int(total_epochs * steps_per_epoch)
           for step in xrange(int(start), training_iter):
               start_time = time.time()
               sess.run(train_op)
               _, loss_value = sess.run([train_op, loss])
               duration = time.time() - start_time
               assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

               if step % 10 == 0:
                   examples_per_sec = self.batch_size / duration
                   sec_per_batch = float(duration)
                   current_epoch = int( step / steps_per_epoch)
                   estimated_time_left = duration * (training_iter - step)/3600
                   format_str = ('step %d. Epoch %d out of %d. loss = %.2f (%.1f examples/sec; %.3f '
                   'sec/batch). Estimated time left: %.4f hours')
                   print(format_str % (step,current_epoch,total_epochs, loss_value, examples_per_sec, sec_per_batch, estimated_time_left))
               if step % 100 == 0:
                   summary_str = sess.run(summary_op)
                   summary_writer.add_summary(summary_str, step)
               # Save the model checkpoint periodically.
               if step % 1000 == 0 or (step + 1) == training_iter:
                   checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
                   saver.save(sess, checkpoint_path, global_step=step)

    def fit(self, coinlabel, total_epochs = 25, grade = True, use_logit = False,load_save = False, do = True, balance_classes = False, weight_decay = .0004):
    #name labels say Grade = False
      with tf.Graph().as_default():
        #weight the classes for inbalance puproses
        global_step = tf.Variable(0, trainable=False)
        if not balance_classes:
            class_weights = tf.constant(coinlabel.get_class_weights(), tf.float32)
        # Build a Graph that computes the logits predictions from the
        # inference model.
            feature_batch, grade_batch, name_batch = tfinput.input(coinlabel.get_file_list(test = False), self.batch_size)
        else:
            feature_batch, grade_batch, name_batch = tfinput.input(coinlabel.get_balanced_class_filelist(test = False, num_per_class= 25000), self.batch_size)
        logits = self.encoding(feature_batch, coinlabel.n_labels, do, self.batch_size, weight_decay)
        if not balance_classes:
            logits = tf.mul(logits, class_weights)
        if use_logit:
            logit_pred, logit_cost = self._add_logit(grade_batch,name_batch, coinlabel.n_names, coinlabel.n_grades)
            logits =   tf.mul(logits, logit_pred)
        # Calculate loss.
        if grade:
            loss = tf_helpers.loss(logits, grade_batch)
        else:
            loss = tf_helpers.loss(logits, name_batch)
        if use_logit:
            loss = loss + logit_cost
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = self._train(loss, global_step)
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(self.save_dir, sess.graph)
        ### pool the queueing processed
        sess.run(init)
        if load_save:
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                start = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            start = 0
        tf.train.start_queue_runners(sess=sess)
        steps_per_epoch = int(len(coinlabel.get_file_list(False))/self.batch_size)
        training_iter = int(total_epochs * steps_per_epoch)
        for step in xrange(int(start), training_iter):
            start_time = time.time()
            sess.run(train_op)
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = self.batch_size / duration
                sec_per_batch = float(duration)
                current_epoch = int( step / steps_per_epoch)
                estimated_time_left = duration * (training_iter - step)/3600
                format_str = ('step %d. Epoch %d out of %d. loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch). Estimated time left: %.4f hours')
                print(format_str % (step,current_epoch,total_epochs, loss_value, examples_per_sec, sec_per_batch, estimated_time_left))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == training_iter:
                checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    def _train(self, total_loss, global_step, learning_rate = .001):
        # Generate moving averages of all losses and associated summaries.

        loss_averages_op = tf_helpers.add_loss_summaries(total_loss)
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(learning_rate)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
          # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

          # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

          # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
              self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

    def _add_logit(self, grade_batch,name_batch, num_names, num_grades):
        one_hot_names = tf.one_hot(name_batch, num_names,  axis = 1)
        W = tf.Variable(tf.zeros([num_names, num_grades]))
        b = tf.Variable(tf.zeros([num_grades]))
        pred = tf.nn.softmax(tf.matmul(one_hot_names, W) + b, name = "wide_prediction") # Softmax
        if grade_batch is not None:
            one_hot_grades = tf.one_hot(grade_batch, num_grades,  axis = 1)
            cost = tf.reduce_mean(-tf.reduce_sum(one_hot_grades*tf.log(pred), reduction_indices=1), name = "wide_cross_entropy")
            return pred, cost
        return pred

    def evaluate(self, coinlabel, grade = True, over_fit_test = False, use_logit = False):
      with tf.Graph().as_default() as g:

        if over_fit_test:
            feature_batch, grade_batch, name_batch = tfinput.input(coinlabel.get_overfit_test_list(), self.batch_size)
        else:
            feature_batch, grade_batch, name_batch = tfinput.input(coinlabel.get_file_list(test = True), self.batch_size)
        logits = self.encoding(feature_batch, coinlabel.n_labels, do = False)
        #find top k predictions
        if use_logit:
                logit_pred = self._add_logit(None, name_batch, coinlabel.n_names, coinlabel.n_grades)
                logits =   tf.mul(logits, logit_pred)
        if grade:
            top_k_op = tf.nn.in_top_k(logits, grade_batch, 1)
        else:
            top_k_op = tf.nn.in_top_k(logits, name_batch, 1)
        # variable_averages = tf.train.ExponentialMovingAverage(
                                    self.moving_average_decay)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.save_dir, g)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
                # Restore the moving average version of the learned variables for eval.
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                     start=True))
                if over_fit_test:
                    num_iter = 20
                else:
                    num_iter = int(math.ceil(len(coinlabel.test_df) / self.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * self.batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                print step
                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            return precision

    def predict_images(self, file_list, conversion_prop, name_lbls = None, n_labels = 4):
      #requires a list of same len as file_list for implementing a wide and deep model
      with tf.Graph().as_default() as g:
        feature_batch, _, name_batch = tfinput.input(file_list, make_coins = True, name_lbls = name_lbls,
                                                                coin_prop = conversion_prop, batch_size = len(file_list))
        logits = self.encoding(feature_batch, n_labels, batch_size = 1, do = False, weight_decay = 0)
        if name_lbls is not None:
            logit_pred = self._add_logit(None,name_batch, num_names = 60, num_grades = 4)
            logits =   tf.mul(logits, logit_pred)
        predict = tf.nn.top_k(logits)

        #find top k predictions
        variable_averages = tf.train.ExponentialMovingAverage(
                                        self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # extract global_step from it.
            else:
                print('No checkpoint file found')
                return
            coord = tf.train.Coordinator()
                # Restore the moving average version of the learned variables for eval.
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                     start=True))
                predictions = sess.run([predict])

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            predictions = sess.run([predict])
      for f in map(lambda name: name + '_tmp', file_list):
          os.remove(f)
      return map(lambda item: item.indices[0,0], predictions)
