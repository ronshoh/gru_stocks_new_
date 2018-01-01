from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


class Model(object):
    """A Variational RHN model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.size = size = config.hidden_size
        self.num_layers = num_layers = config.num_layers
        self.num_of_features = num_of_features = config.num_of_features

        self.in_size = rhn_in_size = size

        self.out_size = out_size = config.out_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, num_of_features])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._mask = tf.placeholder(tf.float32, [batch_size, num_steps])
        # self._noise_x = tf.placeholder(tf.float32, [batch_size, num_steps, 1])
        self._noise_i = tf.placeholder(tf.float32, [batch_size, rhn_in_size, num_layers])
        self._noise_h = tf.placeholder(tf.float32, [batch_size, size, num_layers])
        self._noise_o = tf.placeholder(tf.float32, [batch_size, 1, size])

        inputs = tf.reshape(self._input_data, [-1, num_of_features])

        with tf.variable_scope('w_in'):
            w_in_mat = tf.get_variable("W_in_mat", [num_of_features, size])
            w_in_b = tf.get_variable("softmax_b", [size])
            inputs = tf.matmul(inputs, w_in_mat) + w_in_b

        inputs = tf.reshape(inputs, [batch_size, num_steps, size])
        outputs = []
        self._initial_state = [0] * self.num_layers
        state = [0] * self.num_layers
        self._final_state = [0] * self.num_layers
        for l in range(config.num_layers):
            with tf.variable_scope('RHN' + str(l)):
                if config.layer_norm:
                    cell = GRU(size)
                else:
                    cell = GRU(size, normalizer=None)
                self._initial_state[l] = cell.zero_state(batch_size, tf.float32)
                state[l] = self._initial_state[l]
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state[l]) = cell(inputs[:, time_step, :], state[l],
                                                   [self._noise_i[:, :, l], self._noise_h[:, :, l]])
                    outputs.append(cell_output)
                inputs = tf.stack(outputs, axis=1)
                outputs = []

        output = tf.reshape(inputs * self._noise_o, [-1, size])

        w_out_mat =  tf.get_variable("w_out_mat", [size, out_size])
        b_out_mat = tf.get_variable("b_out_mat", [out_size], initializer=tf.zeros_initializer())

        scores = tf.matmul(output, w_out_mat) + b_out_mat

        self._predictions = tf.reshape(scores, [batch_size, num_steps])

        loss = tf.losses.mean_squared_error(
            self._targets,
            self._predictions,
            weights=self._mask,
            scope="MSE"
        )


        pred_loss = tf.reduce_sum(loss) # / (tf.reduce_sum(self._mask) + epsilon)

        self._cost = cost = pred_loss

        self._final_state = [s for s in state]

        if not is_training:
            self._global_norm = tf.constant(0.0, dtype=tf.float32)
            self._l2_loss = tf.constant(0.0, dtype=tf.float32)
            return

        self._tvars = tf.trainable_variables()
        self._l2_loss = l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._tvars])
        self._cost = cost = pred_loss + config.weight_decay * l2_loss

        self._lr = tf.Variable(0.0, trainable=False)
        self._nvars = np.prod(self._tvars[0].get_shape().as_list())
        print(self._tvars[0].name, self._tvars[0].get_shape().as_list())
        for var in self._tvars[1:]:
            sh = var.get_shape().as_list()
            print(var.name, sh)
            self._nvars += np.prod(sh)
        print(self._nvars, 'total variables')
        grads, self._global_norm = tf.clip_by_global_norm(tf.gradients(cost, self._tvars),
                                          config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, self._tvars))

        with tf.variable_scope('optimizers'):
            if config.adaptive_optimizer == "Adam":
                optimizer_ad = tf.train.AdamOptimizer(self.lr)
            elif config.adaptive_optimizer == "RMSProp":
                optimizer_ad = tf.train.RMSPropOptimizer(self.lr)
            else:
                print("invalid optimizer option.. exiting!")
                optimizer_ad = []
                exit()
            self._train_op_ad = optimizer_ad.apply_gradients(zip(grads, self._tvars))

            with tf.variable_scope('ASGD'):
                self._counter = tf.Variable(0.0, trainable=False)
                optimizer_sgd = tf.train.GradientDescentOptimizer(self.lr)

                self._final_weights = []
                self._temp_weights = []
                for var in self._tvars:
                    self._final_weights.append(tf.get_variable(var.op.name + '_final',
                                                               initializer=tf.zeros_like(var, dtype=tf.float32),
                                                               trainable=False))
                    self._temp_weights.append(tf.get_variable(var.op.name + '_temp',
                                                              initializer=tf.zeros_like(var, dtype=tf.float32),
                                                              trainable=False))



                self._train_op_sgd = optimizer_sgd.apply_gradients(zip(grads, self._tvars))

                adder = tf.Variable(1.0, trainable=False)
                self._add_counter_op = tf.assign_add(self._counter, adder)
                self._asgd_acc_op = [tf.assign_add(self._final_weights[i], var) for i, var in
                                     enumerate(self._tvars)]
                self._reset_accumulation_op = [tf.assign(self._final_weights[i], tf.zeros_like(var)) for i, var in
                                               enumerate(self._final_weights)]

                self._set_asgd_weights = [tf.assign(self._tvars[i], tf.divide(var, self._counter)) for i, var
                                          in enumerate(self._final_weights)]
                self._store_weights = [tf.assign(self._temp_weights[i], var) for i, var in enumerate(self._tvars)]

                self._return_regular_weights = [tf.assign(self._tvars[i], var) for i, var
                                                in enumerate(self._temp_weights)]

    def reset_asgd(self, session):
        counter = session.run(self.counter)
        session.run(tf.assign(self.counter, counter * 0))
        session.run(self.reset_accumulation_op)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def store_set_asgd_weights(self, session):
        session.run(self.store_weights)
        session.run(self.set_asgd_weights)

    @property
    def add_counter_op(self):
        return self._add_counter_op

    @property
    def asgd_acc_op(self):
        return self._asgd_acc_op

    @property
    def return_regular_weights(self):
        return self._return_regular_weights

    @property
    def reset_accumulation_op(self):
        return self._reset_accumulation_op

    @property
    def store_weights(self):
        return self._store_weights

    @property
    def set_asgd_weights(self):
        return self._set_asgd_weights

    @property
    def counter(self):
        return self._counter

    @property
    def final_weights(self):
        return self._final_weights

    @property
    def temp_weights(self):
        return self._temp_weights

    @property
    def tvars(self):
        return self._tvars

    @property
    def predictions(self):
        return self._predictions

    @property
    def l2_loss(self):
        return self._l2_loss

    @property
    def global_norm(self):
        return self._global_norm

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def mask(self):
        return self._mask

    @property
    def noise_i(self):
        return self._noise_i

    @property
    def noise_h(self):
        return self._noise_h

    @property
    def noise_o(self):
        return self._noise_o

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op_ad(self):
        return self._train_op_ad

    @property
    def train_op_sgd(self):
        return self._train_op_sgd


    @property
    def nvars(self):
        return self._nvars


class GRU(tf.contrib.rnn.RNNCell):

    def __init__(
            self, size, activation=tf.tanh, reuse=None,
            normalizer=tf.contrib.layers.layer_norm,
            initializer=tf.contrib.layers.xavier_initializer()):
        super(GRU, self).__init__(_reuse=reuse)
        self._size = size
        self._activation = activation
        self._normalizer = normalizer
        self._initializer = initializer

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, input_, state, noise):
        noise_i = noise[0]
        noise_h = noise[1]
        update, reset = tf.split(self._forward(
            'update_reset', [state * noise_h, input_ * noise_i], 2 * self._size, tf.nn.sigmoid,
            bias_initializer=tf.constant_initializer(-1.)), 2, 1)
        candidate = self._forward(
            'candidate', [reset * state * noise_h, input_ * noise_i], self._size, self._activation)
        state = (1 - update) * state + update * candidate
        return state, state

    def _forward(self, name, inputs, size, activation, **kwargs):
        with tf.variable_scope(name):
            return _forward(
                inputs, size, activation, normalizer=self._normalizer,
                weight_initializer=self._initializer, **kwargs)


def _forward(
        inputs, size, activation, normalizer=tf.contrib.layers.layer_norm,
        weight_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer()):
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    shapes, outputs = [], []
    # Map each input to individually normalize their outputs.
    for index, input_ in enumerate(inputs):
        shapes.append(input_.shape[1: -1].as_list())
        input_ = tf.contrib.layers.flatten(input_)
        weight = tf.get_variable(
            'weight_{}'.format(index + 1), (int(input_.shape[1]), size),
            tf.float32, weight_initializer)
        output = tf.matmul(input_, weight)
        if normalizer:
            output = normalizer(output)
        outputs.append(output)
    output = sum(outputs)
    # Add bias.
    bias = tf.get_variable(
        'weight', (size,), tf.float32, bias_initializer)
    output += bias
    # Activation function.
    if activation:
        output = activation(output)
    # Restore shape dimensions that are consistent among inputs.
    dim = 0
    while dim < min(len(shape) for shape in shapes):
        none = shapes[0].as_list()[dim]
        equal = all(shape[dim] == shapes[0][dim] for shape in shapes)
        if none or not equal:
            break
        dim += 1
    shape = output.shape.as_list()[:1] + shapes[0][:dim] + [-1]
    output = tf.reshape(output, shape)
    return output
