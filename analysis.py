import os

import tensorflow as tf

import inference
from tensorflow.examples.tutorials.mnist import input_data
from data_provider import DataSet

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model/tensorflow"
MODEL_NAME = "tensorflow_model.ckpt"

sess_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.3
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

# server = tf.train.Server.create_local_server(config=sess_config)

session_opts = {
    # 'target': server.target,
    'config': sess_config
}


def train(datasource):
    x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # y = inference.inference(x, regularizer)

    layer_define = {
        'activation_fn': tf.nn.relu,
        'normalizer_fn': None,
        'normalizer_params': None,
        'weights_initializer': tf.contrib.layers.xavier_initializer(),
        'weights_regularizer': regularizer,
        'biases_initializer': tf.zeros_initializer(),
        'biases_regularizer': None,
        'reuse': None,
        'variables_collections': None,
        'outputs_collections': None,
        'trainable': True,
        'scope': None
    }

    y_last = inference.make_layer(x, 3, inference.LAYER_NODE, **layer_define)
    y = inference.make_layer(y, 1, inference.OUTPUT_NODE, **layer_define)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        datasource.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session(**session_opts) as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = datasource.next_batch(BATCH_SIZE)
            if xs is None or len(xs.shape) != 2:
                print(xs)
                continue
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    datasource = DataSet()
    # datasource = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(datasource)


if __name__ == '__main__':
    tf.app.run()
