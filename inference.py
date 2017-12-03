import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from functools import partial

INPUT_NODE = 400

# OUTPUT_NODE = 87474

OUTPUT_NODE = 10000

LAYER_NODE = 10


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference_multi(input_tensor, regularizer, num):
    layer = input_tensor
    for i in range(num):
        in_node = INPUT_NODE if i == 1 else LAYER_NODE
        out_node = OUTPUT_NODE if i == num - 1 else LAYER_NODE
        with tf.variable_scope('layer%d' % (i + 1)):
            weights = get_weight_variable(
                [in_node, out_node],
                regularizer)
            biases = tf.get_variable("biases", [out_node], initializer=tf.constant_initializer(0.0))
            layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    return layer


def make_layer(inputs, layer_num, out_num, **kwargs):
    # y = inputs
    # for i in range(layer_num):
    y = fully_connected(inputs, out_num, **kwargs)

    return y

    # fully_connected(
    #     input_tensor=inputs,
    #     num_outputs=out_num,
    #     **kwargs
    # )

    # fully_connected(
    #     input_tensor,
    #     num_outputs,
    #     activation_fn=tf.nn.relu,
    #     normalizer_fn=None,
    #     normalizer_params=None,
    #     weights_initializer=initializers.xavier_initializer(),
    #     weights_regularizer=None,
    #     biases_initializer=tf.zeros_initializer(),
    #     biases_regularizer=None,
    #     reuse=None,
    #     variables_collections=None,
    #     outputs_collections=None,
    #     trainable=True,
    #     scope=None
    # )


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
