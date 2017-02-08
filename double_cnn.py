"""
Implementation of double CNN.

See https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py
and
https://github.com/Shuangfei/doublecnn/blob/master/main.py#L43-L78

Publication:
    https://arxiv.org/pdf/1610.09716v1.pdf - especially "Algorithm 1" on page 5
"""


import tensorflow as tf
from tflearn import utils
from tflearn import losses, initializations, variables as vs, activations
import numpy as np


def conv_2d_double(incoming, nb_filter, filter_size, strides=1, padding='same',
                   activation='linear', bias=True,
                   weights_init='uniform_scaling',
                   bias_init='zeros', regularizer=None, weight_decay=0.001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="Conv2D"):
    """Double convolution."""
    input_shape = utils.get_incoming_shape(incoming)
    print("input_shape: %s" % str(input_shape))  # None x w^l x h^l x c^l
    # nb_filter = c^{l+1}
    # filter_size = z'
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # The double conovlution part
        # Try 1: Implement a simplified version of the pseudo-code
        z = 3  # z' = z?
        c = input_shape[-1]  # c^l
        c_n = nb_filter  # c^{l+1}
        I = np.eye(c * z**2, dtype=np.float32)
        I = I.reshape(z, z, c * z**2, c)
        w = input_shape[1]  # for now, w^l = w^{l+1} - only padding='same'
        h = input_shape[2]  # for now, w^l = w^{l+1} - only padding='same'
        W_tilde = tf.nn.conv2d(W, I, strides, padding='VALID')
        # ValueError: Dimensions must be equal, but are 32 and 9 for
        # 'Conv2D/Conv2D' (op: 'Conv2D') with input shapes:
        # [3,3,1,32], [3,3,9,1].

        W_tilde = W_tilde.reshape(c_n, c, z, z)
        O_n = tf.nn.conv2d(I, W_tilde, strides, padding='VALID')
        O_n = O_n.reshape(c_n * w * h, 1, 1)
        I_n = O_n  # for now, stride s =1 -> no pooling
        inference = I_n.reshape(c_n, w, h)

        # Try 2: Translate the theano code:
        # filter_offset = filter_size[0] - kernel_size + 1
        # n_times = filter_offset ** 2
        # W_shape = (nb_filter * n_times, input_shape[1]) + (kernel_size,) * 2
        # #           (32      *       1, 28,               (3, ) )
        # print("W_shape: %s" % str(W_shape))
        # prod_ = np.prod(W_shape[1:])
        # print("prod: %s" % prod_)
        # identity = np.eye(prod_, dtype=np.float32)  # I_l \in c^l z^2
        # new_shape = (np.prod(W_shape[1:]),) + W_shape[1:]
        # new_shape = (new_shape[2],
        #              new_shape[3],
        #              new_shape[0],
        #              new_shape[1],)
        # print("identitysss: %s" % str(identity.shape))
        # print("new shape: %s" % str(new_shape))
        # filter_ = np.reshape(identity, new_shape)

        # W_effective = tf.nn.conv2d(W, filter_, strides, padding='VALID')
        # inference = tf.nn.conv2d(incoming, W_effective, strides, padding)

        # Normal convolution again
        if b:
            inference = tf.nn.bias_add(inference, b)

        if activation:
            if isinstance(activation, str):
                inference = activations.get(activation)(inference)
            elif hasattr(activation, '__call__'):
                inference = activation(inference)
            else:
                raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference
