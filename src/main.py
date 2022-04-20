"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import argparse
import itertools
import logging.config

import numpy as np
import tensorflow as tf
import tqdm

from util import ranking, reader, EarlyStopping


def args():
    parser = argparse.ArgumentParser(description='Continuous-Time Self-Modulating Attention (CTSMA)')
    parser.add_argument('--train', action="store", required=True,
                        help="training data file patterns", )
    parser.add_argument('--test', action="store", required=True,
                        help="test data file patterns")
    parser.add_argument('--model', action="store", required=True,
                        help="algorithm names")
    parser.add_argument('--num_items', type=int, required=True)

    # ---- for definition
    parser.add_argument('--num_units', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--seqslen', type=int, default=30)
    parser.add_argument('--timelen', type=int, default=256)

    # ---- for SMACTREC
    parser.add_argument('--mark', type=str, help="mark data file")
    parser.add_argument('--ct_reg', type=float, default=0.)

    # ---- for optimization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--l2_reg', type=float)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.)
    parser.add_argument('--attention_probs_dropout_rate', type=float, default=0.)
    parser.add_argument('--num_train_steps', type=int)
    parser.add_argument('--num_warmup_steps', type=int)

    # ---- for evaluation
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--mask_seen', action="store_true", dest="mask_seen")
    parser.set_defaults(l2_reg=0., mask_seen=False)
    return parser.parse_args()


def main():
    logging.info("1. build data pipeline")
    with tf.device('/cpu:0'):
        tr_reader = reader(FLAGS, FLAGS.train, is_training=True)
        tr_data = tr_reader(FLAGS.batch_size).make_initializable_iterator()

        te_reader = reader(FLAGS, FLAGS.test, is_training=False)
        te_data = te_reader(FLAGS.batch_size).make_initializable_iterator()

    logging.info("2. create neural model")
    with tf.variable_scope("main"):
        m = ranking(FLAGS)
        features, labels = tr_data.get_next()
        train_op, loss_op, loss_init_op = m.train(features, labels)

        # reuse variables for the next tower.
        tf.get_variable_scope().reuse_variables()

        features, labels = te_data.get_next()
        metrics_op, metric_init_op = m.eval(features, labels, mask_seen=FLAGS.mask_seen)

    logging.info("3. train and evaluate model")
    stopper = EarlyStopping(FLAGS)
    training_steps = 0

    config = tf.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch in range(FLAGS.num_epochs):
            sess.run([loss_init_op, tr_data.initializer])
            with tqdm.tqdm(itertools.count(), ascii=True) as tq:
                try:
                    for step in tq:
                        training_steps += 1

                        _, running_loss = sess.run([train_op, loss_op])
                        if step % 10 == 0:
                            tq.set_postfix(loss='{0:.5f}'.format(running_loss))
                except tf.errors.OutOfRangeError:
                    logging.info("%03d: Loss=%.4f" % (epoch, running_loss))

            sess.run([metric_init_op, te_data.initializer])
            with tqdm.tqdm(itertools.count(), ascii=True) as tq:
                try:
                    for _ in tq:
                        metrics = sess.run(metrics_op)
                except tf.errors.OutOfRangeError:
                    logging.info("%03d: %s" % (epoch, {k: "{0:.5f}".format(v) for k, v in metrics.items()}))

            stopper.step(running_loss, metrics['H10'])  # focused on HR[changable]
            # stopping when no performance gain is achieved
            if stopper.early_stop:
                break


if __name__ == "__main__":
    logging.config.fileConfig('./conf/logging.conf')
    np.random.seed(9876)
    tf.random.set_random_seed(9876)

    FLAGS = args()
    logging.info("================")
    logging.info(vars(FLAGS))
    logging.info("================")

    main()
