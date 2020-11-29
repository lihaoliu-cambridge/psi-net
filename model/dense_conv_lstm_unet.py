from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.upsample import linear_upsample_3d
from dltk.core.activations import leaky_relu

from tensorflow.contrib import rnn
from model.residual_unit import vanilla_residual_unit_3d
from model.layers.dense_conv_lstm_layer import Conv3DLSTMCell

global_conv_params = {'padding': 'same',
                      'use_bias': False,
                      'kernel_initializer': tf.initializers.variance_scaling(distribution='uniform'),
                      'bias_initializer': tf.zeros_initializer(),
                      'kernel_regularizer': tf.contrib.layers.l2_regularizer(1e-4),
                      'bias_regularizer': None}


def upsample(inputs, strides=(2, 2, 2)):
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    # Upsample inputs
    inputs = linear_upsample_3d(inputs, strides)

    return inputs


def upsample_and_conv(inputs, strides=(2, 2, 2), mode=tf.estimator.ModeKeys.EVAL, conv_params=global_conv_params, name=None):
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    # Upsample inputs
    with tf.variable_scope('reduce_channel_unit_{}'.format(name)):
        inputs1 = linear_upsample_3d(inputs, strides)

        result = tf.layers.batch_normalization(inputs1, training=mode == tf.estimator.ModeKeys.TRAIN)
        result = leaky_relu(result)

        result = tf.layers.conv3d(
            inputs=result,
            filters=inputs.get_shape().as_list()[-1] / 2,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            **conv_params)

    return result


def upsample_and_concat(inputs, inputs2, strides=(2, 2, 2)):
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), \
        'Ranks of input and input2 differ'

    # Upsample inputs
    inputs1 = linear_upsample_3d(inputs, strides)

    return tf.concat(axis=-1, values=[inputs2, inputs1])


def upsample_and_concat_with_conv(inputs, inputs2, strides=(2, 2, 2), mode=tf.estimator.ModeKeys.EVAL, conv_params=global_conv_params):
    """Upsampling and concatenation layer according to [1].

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.

    Args:
        inputs (TYPE): Input features to be upsampled.
        inputs2 (TYPE): Higher resolution features from the encoder to
            concatenate.
        strides (tuple, optional): Upsampling factor for a strided transpose
            convolution.

    Returns:
        tf.Tensor: Upsampled feature tensor
    """
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), \
        'Ranks of input and input2 differ'

    # Upsample inputs
    inputs1 = linear_upsample_3d(inputs, strides)

    with tf.variable_scope('reduce_channel_unit'):
        result = tf.concat(axis=-1, values=[inputs2, inputs1])
        result = tf.layers.batch_normalization(result, training=mode == tf.estimator.ModeKeys.TRAIN)
        result = leaky_relu(result)

        result = tf.layers.conv3d(
            inputs=result,
            filters=inputs2.shape[-1],
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            **conv_params)

    return result


def residual_unet_3d(inputs,
                     num_classes,
                     num_res_units=1,
                     filters=(16, 32, 64, 128),
                     strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                     mode=tf.estimator.ModeKeys.EVAL,
                     use_bias=False,
                     activation=leaky_relu,
                     kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                     bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None):
    """
    Image segmentation network based on a flexible UNET architecture [1]
    using residual units [2] as feature extractors. Downsampling and
    upsampling of features is done via strided convolutions and transpose
    convolutions, respectively. On each resolution scale s are
    num_residual_units with filter size = filters[s]. strides[s] determine
    the downsampling factor at each resolution scale.

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        num_res_units (int, optional): Number of residual units at each
            resolution scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        activation (optional): A function to use as activation function.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.

    Returns:
        dict: dictionary of output tensors

    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Initial convolution with filters[0]
    x = tf.layers.conv3d(inputs=x,
                         filters=filters[0],
                         kernel_size=(3, 3, 3),
                         strides=strides[0],
                         **conv_params)

    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different resolution scales res_scales
    res_scales = []
    saved_strides = []
    for res_scale in range(len(filters)):
        res_scale_list = []
        # Features are downsampled via strided convolutions. These are defined in `strides` and subsequently saved
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):

            x, x1 = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                mode=mode)
        res_scale_list.append(x1)
        res_scale_list.append(x)
        saved_strides.append(strides[res_scale])

        for i in range(1, num_res_units):

            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):

                x, x1 = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    mode=mode)

            res_scale_list.append(x1)
            res_scale_list.append(x)
        res_scales.append(res_scale_list)

        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Upsample and concat layers [1] reconstruct the predictions to higher resolution scales
    state_list = []
    with tf.variable_scope('layer_agg_{}'.format(len(filters) - 1)):
        number = len(filters) - 1
        initial_hidden = tf.zeros_like(x)
        initial_cell = tf.zeros_like(x)

        initial_state = tf.concat([initial_cell, initial_hidden], axis=4)

        input_shape = [x.get_shape().as_list()[1],
                       x.get_shape().as_list()[2],
                       x.get_shape().as_list()[3],
                       x.get_shape().as_list()[4]]

        lstm = Conv3DLSTMCell(input_shape,
                              [3, 3, 3],
                              num_features=filters[number],
                              scope='lstm_{}'.format(number),
                              activation=tf.nn.relu)

        aggregated_feature_list, state = rnn.static_rnn(lstm, res_scale_list, initial_state=initial_state, dtype=tf.float32)

        c, h = tf.split(axis=4, num_or_size_splits=2, value=state)

        new_c = upsample_and_conv(c[:, :, :, :, c.shape[-1]-aggregated_feature_list[-1].shape[-1]:], (1, 2, 2), mode=mode, name="c")
        new_h = upsample_and_conv(h[:, :, :, :, c.shape[-1]-aggregated_feature_list[-1].shape[-1]:], (1, 2, 2), mode=mode, name="h")

        new_state = tf.concat(axis=4, values=[new_c, new_h])

        state_list.append(new_state)


    for res_scale in range(len(filters) - 2, -1, -1):
        res_scale_list = res_scales[res_scale]

        with tf.variable_scope('layer_agg_{}'.format(res_scale)):
            input_shape = [res_scale_list[-1].get_shape().as_list()[1],
                           res_scale_list[-1].get_shape().as_list()[2],
                           res_scale_list[-1].get_shape().as_list()[3],
                           res_scale_list[-1].get_shape().as_list()[4]]

            lstm = Conv3DLSTMCell(input_shape,
                                  [3, 3, 3],
                                  num_features=filters[res_scale],
                                  scope='lstm_{}'.format(res_scale),
                                  activation=tf.nn.relu)

            aggregated_feature, state = rnn.static_rnn(lstm, res_scale_list, initial_state=state_list[-1], dtype=tf.float32)

            if res_scale != 0:
                c, h = tf.split(axis=4, num_or_size_splits=2, value=state)
                new_c = upsample_and_conv(c[..., c.shape[-1]-aggregated_feature[-1].shape[-1]:], (1, 2, 2), mode=mode, name="c")
                new_h = upsample_and_conv(h[..., h.shape[-1]-aggregated_feature[-1].shape[-1]:], (1, 2, 2), mode=mode, name="h")

                new_state = tf.concat(axis=4, values=[new_c, new_h])

                state_list.append(new_state)

        with tf.variable_scope('up_concat_{}'.format(res_scale)):

            x = upsample_and_concat(
                inputs=x,
                inputs2=aggregated_feature[-1],
                strides=saved_strides[res_scale+1])

        for i in range(2):
            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):
                x, _ = vanilla_residual_unit_3d(
                        inputs=x,
                        out_filters=filters[res_scale],
                        strides=(1, 1, 1),
                        mode=mode)
            tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
                res_scale, x.get_shape()))

    outputs['aggregated_feature_0'] = aggregated_feature[0]
    outputs['aggregated_feature_1'] = aggregated_feature[1]
    outputs['aggregated_feature_2'] = aggregated_feature[2]
    outputs['aggregated_feature_3'] = aggregated_feature[3]

    outputs['res_scale_list_0'] = res_scale_list[0]
    outputs['res_scale_list_1'] = res_scale_list[1]
    outputs['res_scale_list_2'] = res_scale_list[2]
    outputs['res_scale_list_3'] = res_scale_list[3]

    # Last convolution
    with tf.variable_scope('last'):

        x = tf.layers.conv3d(inputs=x,
                             filters=num_classes,
                             kernel_size=(1, 1, 1),
                             strides=(1, 1, 1),
                             **conv_params)

    tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x

    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob

        y_ = tf.argmax(x, axis=-1) \
            if num_classes > 1 \
            else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)

        outputs['y_'] = y_

    return outputs
