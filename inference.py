import tensorflow as tf

INPUT_NODE = 400

# OUTPUT_NODE = 87474

OUTPUT_NODE = 40000

LAYER_NODE = 5


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
