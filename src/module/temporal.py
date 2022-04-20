"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow as tf


class SelfModulatingAttention(object):
    """ Implementation of the paper ---
    C Chen, H Geng, N Yang, J Yan, D Xue, J Yu, X Yang.
    Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation.
    ICML 2021.
    """

    def __init__(self, num_units, num_heads, num_events, dropout_rate, scope="modulating_attention"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_events = num_events
        self.dropout_rate = dropout_rate
        self.scope = scope

    def intensity(self, H, intervals, mark_onehot):
        num_units, num_heads, num_events = self.num_units, self.num_heads, self.num_events
        intervals = tf.tile(tf.expand_dims(intervals, axis=-1), [num_heads, 1, 1])

        # E: number of events, C: number of channels, h: number of hidden units
        # N: batch size
        layer_inputs = tf.concat([H, intervals], axis=-1)
        with tf.variable_scope("sequential_temporal_combined"):
            layers_outputs = tf.layers.dense(layer_inputs, num_units // num_heads * num_events,
                                             activation=tf.nn.sigmoid)  # (h*N, T_q,  h/C*E)
            layers_outputs = tf.concat(tf.split(layers_outputs, num_events, axis=2), axis=0)  # (h*N*E, T_q, C/h)

            weight = tf.get_variable("weight", trainable=True, shape=[num_events, num_units // num_heads],
                                     initializer=tf.glorot_uniform_initializer())
            weight = tf.reshape(weight, shape=[num_events, 1, num_units // num_heads, 1])
            weight = tf.tile(weight, [1, tf.shape(H)[0], 1, 1])
            weight = tf.reshape(weight, [num_events * tf.shape(H)[0], num_units // num_heads, 1])  # (h*N*E, h/C, 1)

            scaling = tf.get_variable("scaling", trainable=True, shape=[num_events],
                                      initializer=tf.zeros_initializer())
            scaling = tf.reshape(tf.exp(scaling), shape=[num_events, 1, 1, 1])
            scaling = tf.tile(scaling, [1, tf.shape(H)[0], 1, 1])
            scaling = tf.reshape(scaling, [num_events * tf.shape(H)[0], 1, 1])  # (h*N*E, 1, 1)

            mark_intensity = tf.matmul(layers_outputs, weight) / scaling
            mark_intensity = scaling * tf.log(1. + tf.exp(mark_intensity))  # (h*N*E, T_q, 1)
            mark_intensity = tf.concat(tf.split(mark_intensity, num_events, axis=0), axis=2)  # (h*N, T_q, E)

            mark_intensity_4d = tf.tile(tf.expand_dims(mark_intensity, 2),
                                        [1, 1, tf.shape(H)[1], 1])  # (h*N, T_q, T_k, E)
            mark_onehot_4d = tf.tile(tf.expand_dims(tf.to_float(mark_onehot), 1),
                                     [num_heads, tf.shape(H)[1], 1, 1])  # (h*N, T_q, T_k, E)
            mark_intensity_4d = tf.reduce_sum(mark_intensity_4d * mark_onehot_4d, axis=-1)  # (h*N, T_q, T_k)

        return mark_intensity_4d, mark_intensity

    @classmethod
    def biased_likelihood(cls, mark_intensity, next_mark_onehot, intervals, num_heads):
        if num_heads != 1:
            next_mark_onehot = tf.tile(next_mark_onehot, [num_heads, 1, 1])
            intervals = tf.tile(intervals, [num_heads, 1])

        # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
        # next_mark_onehot: the mark for the next item (h*N, T_q, E)
        mark_intensity *= tf.sign(tf.reduce_sum(next_mark_onehot, axis=2, keepdims=True))
        event_intensity = tf.reduce_sum(mark_intensity * next_mark_onehot, axis=2)

        event_ll = tf.log(tf.where(tf.equal(event_intensity, 0), tf.ones_like(event_intensity), event_intensity))
        event_ll = tf.reduce_sum(event_ll)

        entire_intensity = tf.reduce_sum(mark_intensity, axis=2)
        nu_integral = entire_intensity * intervals * .5
        non_event_ll = tf.reduce_sum(nu_integral)

        num_events = tf.reduce_sum(next_mark_onehot)
        biased_mle = -tf.reduce_sum(event_ll - non_event_ll) / num_events
        return biased_mle

    def __call__(self, queries, keys, masks, intervals, marks, is_training, causality):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            T = tf.layers.dense(keys, num_units, activation=None)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            T_ = tf.concat(tf.split(T, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Weighted sum
            # marked_intensity_4d: marked intensity for every past events (h*N, T_q, T_k)
            # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
            sequential_units = tf.matmul(outputs, T_)  # ( h*N, T_q, C/h)
            marked_intensity_4d, mark_intensity = self.intensity(sequential_units, intervals, marks)

            # Dropouts
            outputs = marked_intensity_4d * outputs
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries[:, :, :num_units]

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

            return outputs, mark_intensity
