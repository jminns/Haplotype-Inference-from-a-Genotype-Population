import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import _transpose_batch_time


class hap_inf:
    def __init__(self, **kwargs):
        self.hap_lens = tf.cast(kwargs['hap_lens'], dtype=tf.int32)
        self.max_hap_len = kwargs['max_hap_len']
        self.len_ancs = kwargs['len_ancs']
        self.temperature1 = kwargs['temperature']
        self.temperature2 = kwargs['temperature']
        self.temperature_prior = kwargs['temperature_prior']
        self.dim_ancs = kwargs['dim_ancs']
        self.batch_size = kwargs['batch_size']
        self.units_firstlstm = kwargs['units_firstlstm']
        self.dense_units = kwargs['dense_units']
        self.units_bilstm = kwargs['units_bilstm']
        self.units_bilstm2 = kwargs['units_bilstm2']
        self.alt_rnn = kwargs['alt_rnn']
        self.dense_intmdt_pred_units = kwargs['dense_intmdt_pred_units']
        self.lmbda_next = kwargs['lmbda_next']

    def prep_templates(self, n_batches):
        template_mat = np.zeros([n_batches, self.batch_size, self.max_hap_len, self.len_ancs])
        for i in range(n_batches):
            for k, j in enumerate(self.hap_lens[i]):
                template_mat[i, k, 0:j, :] = np.ones(j, self.len_ancs)
        return template_mat

    def att_dot(self, query, values):
        logits = tf.matmul(values, tf.transpose(query))
        return tf.nn.softmax(logits=tf.transpose(logits), axis=-1)

    def p1_encoder_rnn(self, input_encoder, reuse, hap_lens, units_bilstm):
        # take out casting when data is prepared with placeholders
        input_encoder = tf.cast(input_encoder, tf.float32)

        with tf.variable_scope('enc_p1', reuse=reuse):
            cell1 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            cell2 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            values, states = tf.nn.bidirectional_dynamic_rnn(inputs=input_encoder, dtype=tf.float32, cell_bw=cell1,
                                                             cell_fw=cell2, sequence_length=tf.cast(hap_lens, tf.int32))
        values = tf.concat(values, 2)
        # extra dense to split off into the two haplotypes
        values = tf.layers.dense(inputs=values, activation=tf.nn.relu, reuse=reuse, units=units_bilstm)
        return values

    def alt_encoder2_rnn(self, dense_intmdt_pred_units, input_encoder, temperature, units_lstm, train, hap_lens, reuse):
        input_encoder=tf.cast(input_encoder,dtype=tf.float32)

        with tf.variable_scope('enc_p2', reuse=reuse):
            # Ancestors
            Anc = tf.get_variable(name='Ancs', shape=[self.len_ancs, self.dim_ancs])

            w_proj = tf.get_variable(shape=[units_lstm, self.dim_ancs], dtype=tf.float32, name='w_proj')
            b_proj = tf.get_variable(shape=[self.dim_ancs], dtype=tf.float32, name='b_proj')

        cell = tf.contrib.rnn.LSTMCell(units_lstm*2)
        inputs = tf.transpose(input_encoder, perm=[1, 0, 2])
        # had to concat these zeros, kind of awkward, not sure why
        inputs = tf.concat([inputs, tf.zeros([1, self.batch_size, tf.shape(inputs)[-1]], dtype=tf.float32)], axis=0)
        output_ta = (tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32))

        # inputs_ta = tf.TensorArray(dynamic_size=False,dtype=tf.float32,size=self.max_hap_len,clear_after_read=False)
        # inputs_ta.unstack(inputs)

        print(input_encoder)
        print(output_ta)
        print(tf.transpose(input_encoder, perm=[1, 0, 2]))
        # take out when using placeholders
        print('here')

        def loop_fn(time, cell_output, cell_state, loop_state):
            print('cell_output {}'.format(cell_output))
            print('cell_state {}'.format(cell_state))
            # print(inputs_ta)

            emit_output = cell_output  # don't care about this one, only care about loop_state in this case because loop_state doesn't have to be same shape as rnn output
            if cell_output is None:  # time == 0
                print('here1')
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                print('here2')
                print(time)
                next_anc = tf.concat(
                    [tf.zeros(shape=[self.batch_size, self.len_ancs * 2], dtype=tf.float32), inputs[[time]]], axis=-1)
                # inputs_ta.read(time) ], axis=-1)
                print('here2.5')
                print('here3')
                next_loop_state = output_ta
            else:
                print('here4')
                next_cell_state = cell_state
                hap_1, hap_2 = tf.split(cell_output, num_or_size_splits=2, axis=-1)
                with tf.variable_scope('enc_p2', reuse=True):
                    pre_next_anc1 = tf.nn.relu(tf.matmul(hap_1, w_proj) + b_proj)
                    pre_next_anc2 = tf.nn.relu(tf.matmul(hap_2, w_proj) + b_proj)
                    print('here5')
                    anc_distribution_h1 = self.att_dot(query=pre_next_anc1, values=Anc)
                    anc_distribution_h2 = self.att_dot(query=pre_next_anc2, values=Anc)
                if train:
                    dist_h1 = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature,
                                                                                   probs=anc_distribution_h1)
                    next_anc_sample_h1 = dist_h1.sample()
                    dist_h2 = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature,
                                                                                   probs=anc_distribution_h2)
                    next_anc_sample_h2 = dist_h2.sample()
                    # next_anc_sample = anc_distribution
                    next_anc = tf.concat([tf.concat([next_anc_sample_h1, next_anc_sample_h2], axis=-1), inputs[[time]]],
                                         axis=-1)
                else:
                    dist_h1 = tf.contrib.distributions.Categorical(probs=anc_distribution_h1)
                    next_anc_sample_h1 = tf.cast(tf.one_hot(dist_h1.sample(), depth=self.len_ancs, axis=-1),dtype=tf.float32)
                    dist_h2 = tf.contrib.distributions.Categorical(probs=anc_distribution_h2)
                    next_anc_sample_h2 = tf.cast(tf.one_hot(dist_h2.sample(), depth=self.len_ancs, axis=-1),dtype=tf.float32)
                    # next_anc_sample = anc_distribution
                    next_anc = tf.concat([tf.concat([next_anc_sample_h1, next_anc_sample_h2], axis=-1), inputs[[time]]],
                                         axis=-1)

                anc_h1 = tf.reduce_sum(tf.reshape(tf.matmul(tf.reshape(tf.matrix_diag(next_anc_sample_h1),[-1,self.len_ancs]), Anc),[self.batch_size,self.len_ancs,self.dim_ancs]),1)
                anc_h2 = tf.reduce_sum(tf.reshape(tf.matmul(tf.reshape(tf.matrix_diag(next_anc_sample_h2),[-1,self.len_ancs]), Anc),[self.batch_size,self.len_ancs,self.dim_ancs]),1)

                anc_h1_2 = tf.layers.dense(anc_h1, units=dense_intmdt_pred_units, activation=tf.nn.relu)
                anc_h2_2 = tf.layers.dense(anc_h2, units=dense_intmdt_pred_units, activation=tf.nn.relu)

                pred_current_h1 = tf.layers.dense(anc_h1_2, units=1, activation=None)
                pred_next_h1 = tf.layers.dense(anc_h1_2, units=1, activation=None)
                pred_current_h2 = tf.layers.dense(anc_h2_2, units=1, activation=None)
                pred_next_h2 = tf.layers.dense(anc_h2_2, units=1, activation=None)



                # this is sent as input to the next iteration of the cell

                # inputs_ta.read(time)], axis=-1)
                print('here7')
                # output to store for the iteration
                next_loop_state = (
                    loop_state[0].write(time - 1, next_anc_sample_h1),
                    loop_state[1].write(time - 1, next_anc_sample_h1), loop_state[2].write(time - 1, pred_current_h1),
                    loop_state[3].write(time - 1, pred_next_h1), loop_state[4].write(time - 1, pred_current_h2),
                    loop_state[5].write(time - 1, pred_next_h2))

            print('out_loop')
            # this gives us a vector in the size of the batch, telling us which elements have finished
            elements_finished = time >= hap_lens
            print(elements_finished)
            # because we are not interested in the state

            return (elements_finished, next_anc, next_cell_state, emit_output, next_loop_state)

        with tf.variable_scope('state', reuse=reuse):
            _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
        print('Anc_O {}'.format(_transpose_batch_time(loop_state_ta[0].stack())))
        print('params_O {}'.format(_transpose_batch_time(loop_state_ta[1].stack())))

        X_sampled_h1 = _transpose_batch_time(loop_state_ta[0].stack())
        X_sampled_h2 = _transpose_batch_time(loop_state_ta[1].stack())
        reconstruction_h1 = tf.nn.sigmoid(_transpose_batch_time(loop_state_ta[2].stack()))
        pred_next_rec_h1 = tf.nn.sigmoid(_transpose_batch_time(loop_state_ta[3].stack())[:, 0:-1])
        reconstruction_h2 = tf.nn.sigmoid(_transpose_batch_time(loop_state_ta[4].stack()))
        pred_next_rec_h2 = tf.nn.sigmoid(_transpose_batch_time(loop_state_ta[5].stack())[:, 0:-1])

        ####DONT FORGET TO CUT OFF LAST next allele PREDICTION, MEANINGLESS

        return X_sampled_h1, X_sampled_h2, reconstruction_h1, pred_next_rec_h1, reconstruction_h2, pred_next_rec_h2

    def p2_encoder2_rnn(self, input_encoder, temperature, units_lstm, train, hap_lens, reuse):
        with tf.variable_scope('enc_p2', reuse=reuse):
            # Ancestors
            Anc = tf.get_variable(name='Ancs', shape=[self.len_ancs, self.dim_ancs])

            w_proj = tf.get_variable(shape=[units_lstm, self.dim_ancs], dtype=tf.float32, name='w_proj')
            b_proj = tf.get_variable(shape=[self.dim_ancs], dtype=tf.float32, name='b_proj')

        cell = tf.contrib.rnn.LSTMCell(units_lstm)
        inputs = tf.transpose(input_encoder, perm=[1, 0, 2])
        # had to concat these zeros, kind of awkward, not sure why
        inputs = tf.concat([inputs, tf.zeros([1, self.batch_size, tf.shape(inputs)[-1]], dtype=tf.float32)], axis=0)
        output_ta = (tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32),
                     tf.TensorArray(size=self.max_hap_len, dtype=tf.float32))

        # inputs_ta = tf.TensorArray(dynamic_size=False,dtype=tf.float32,size=self.max_hap_len,clear_after_read=False)
        # inputs_ta.unstack(inputs)

        print(input_encoder)
        print(output_ta)
        print(tf.transpose(input_encoder, perm=[1, 0, 2]))
        # take out when using placeholders
        print('here')

        def loop_fn(time, cell_output, cell_state, loop_state):
            print('cell_output {}'.format(cell_output))
            print('cell_state {}'.format(cell_state))
            # print(inputs_ta)

            emit_output = cell_output  # don't care about this one, only care about loop_state in this case because loop_state doesn't have to be same shape as rnn output
            if cell_output is None:  # time == 0
                print('here1')
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                print('here2')
                print(time)
                next_anc = tf.concat([tf.zeros(shape=[self.batch_size, self.len_ancs], dtype=tf.float32), inputs[[time]]], axis=-1)
                # inputs_ta.read(time) ], axis=-1)
                print('here2.5')
                print('here3')
                next_loop_state = output_ta
            else:
                print('here4')
                next_cell_state = cell_state
                with tf.variable_scope('enc_p2', reuse=True):
                    pre_next_anc1 = tf.nn.relu(tf.matmul(cell_output, w_proj) + b_proj)
                    print('here5')
                    anc_distribution = self.att_dot(query=pre_next_anc1, values=Anc)
                if train:
                    dist = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature,
                                                                                probs=anc_distribution)
                    next_anc_sample = dist.sample()
                    # next_anc_sample = anc_distribution
                    next_anc = tf.concat([next_anc_sample, inputs[[time]]], axis=-1)
                else:
                    dist = tf.contrib.distributions.Categorical(probs=anc_distribution)
                    next_anc_sample = tf.cast(tf.one_hot(dist.sample(), depth=self.len_ancs, axis=-1), dtype=tf.float32)
                    # next_anc_sample=anc_distribution
                    next_anc = tf.concat([next_anc_sample, inputs[[time]]], axis=-1)

                print('ANC DIST {}'.format(anc_distribution))
                print('here6')

                print(next_anc_sample)

                # this is sent as input to the next iteration of the cell

                # inputs_ta.read(time)], axis=-1)
                print('here7')
                # output to store for the iteration
                next_loop_state = (
                loop_state[0].write(time - 1, next_anc_sample), loop_state[1].write(time - 1, anc_distribution),
                loop_state[2].write(time - 1, pre_next_anc1))

            print('out_loop')
            # this gives us a vector in the size of the batch, telling us which elements have finished
            elements_finished = time >= hap_lens
            print(elements_finished)
            # because we are not interested in the state
            print('next_anc {}'.format(next_anc))
            print('next_cell {}'.format(next_cell_state))
            return (elements_finished, next_anc, next_cell_state, emit_output, next_loop_state)

        with tf.variable_scope('state', reuse=reuse):
            _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
        print('Anc_O {}'.format(_transpose_batch_time(loop_state_ta[0].stack())))
        print('params_O {}'.format(_transpose_batch_time(loop_state_ta[1].stack())))

        X_sampled = _transpose_batch_time(loop_state_ta[0].stack())
        dist_params = _transpose_batch_time(loop_state_ta[1].stack())
        query_vecs = _transpose_batch_time(loop_state_ta[2].stack())

        return X_sampled, dist_params, query_vecs

    def p2_encoder_rnn(self, input_encoder, units_firstbilstm, reuse):

        with tf.variable_scope('enc_p2', reuse=reuse):
            # Ancestors
            Anc = tf.get_variable(name='Ancs', shape=[self.len_ancs, self.dim_ancs])
            cell = tf.contrib.rnn.BasicLSTMCell(1024)
            w_proj = tf.get_variable(shape=[1024, self.dim_ancs], dtype=tf.float32, name='w_proj')
            b_proj = tf.get_variable(shape=[self.dim_ancs], dtype=tf.float32, name='b_proj')
        print(cell)

        y0 = tf.zeros([self.batch_size, self.len_ancs], dtype=tf.float32)
        # y_z = tf.concat([y0, input_encoder[:,0,:]], axis=-1)
        y_z = y0
        h2 = tf.zeros([self.batch_size, 1, self.len_ancs], dtype=tf.float32)
        i0 = tf.constant(0)
        # might have to make this lstm tuple, initialization is meaningless, will change in loop
        s0_shape = tf.contrib.rnn.LSTMStateTuple(tf.TensorShape([None, 1024]), tf.TensorShape([None, 1024]))
        s0 = tf.contrib.rnn.LSTMStateTuple(tf.zeros([self.batch_size, 1024]),
                                           tf.zeros([self.batch_size, 1024], dtype=tf.float32))
        time_steps = self.max_hap_len

        def c(i, s0, h2, y_z):
            return i < time_steps

        def b1(i, s0, h2, y_z):
            print(input_encoder)
            print(y_z)
            print("here")
            y_z = tf.concat([y_z, input_encoder[:, i, :]], axis=-1)
            print("here2")
            print("here3")
            print(y_z)
            y_z = tf.reshape(y_z, shape=[self.batch_size, (self.len_ancs + units_firstbilstm * 2)])
            print(y_z)
            print("here4")
            outputs_1, state = cell(y_z, s0)
            print("here5")
            outputs_2 = tf.nn.relu(tf.matmul(outputs_1, w_proj) + b_proj)
            anc_distribution = self.att_dot(query=outputs_2, values=Anc)
            dist = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=self.temperature,
                                                                        probs=anc_distribution)
            anc_sample = dist.sample()
            print('h2 {} anc_sample {}'.format(h2, anc_sample))
            return [i + 1, state, tf.concat([h2, tf.reshape(anc_sample, [self.batch_size, 1, self.len_ancs])], axis=1),
                    anc_sample]

        with tf.variable_scope('enc_p2', reuse=True):
            ii, s0, h2, y_z = tf.while_loop(parallel_iterations=1, cond=c, body=b1, loop_vars=[i0, s0, h2, y_z],
                                            shape_invariants=[i0.get_shape(), s0_shape,
                                                              tf.TensorShape([self.batch_size, None, self.len_ancs]),
                                                              y_z.get_shape()])
        h2 = tf.slice(h2, [0, 1, 0], [-1, -1, -1])
        return h2, y_z

    def run_alt_enc(self, train, units_firstlstm, batch_a_haps, dense_intmdt_pred_units, hap_lens, reuse):
        X_sampled_h1, X_sampled_h2, reconstruction_h1, pred_next_rec_h1, reconstruction_h2, pred_next_rec_h2 = self.alt_encoder2_rnn(
            dense_intmdt_pred_units=dense_intmdt_pred_units, input_encoder=batch_a_haps, temperature=self.temperature1,
            units_lstm=units_firstlstm, train=train, hap_lens=hap_lens, reuse=reuse)
        return X_sampled_h1, X_sampled_h2, reconstruction_h1, pred_next_rec_h1, reconstruction_h2, pred_next_rec_h2

    def compute_alt_loss(self, test, ort_constr, lmbda_next, genotypes, lat_h1, lat_h2, pred_rec_h1, pred_rec_h2,
                         pred_next_h1, pred_next_h2, constr_q=False):
        next_step_genotypes = genotypes[:, 1:]

        reconstruction_p1 = tf.log(tf.concat(
            [(1 - pred_rec_h1) * (1 - pred_rec_h2), (1 - pred_rec_h2) * pred_rec_h1 + pred_rec_h2 * (1 - pred_rec_h1),
             pred_rec_h1 * pred_rec_h2], axis=-1) + 1e-9)
        reconstruction_next_p1 = tf.log(tf.concat([(1 - pred_next_h1) * (1 - pred_next_h2),
                                                   (1 - pred_next_h2) * pred_next_h1 + pred_next_h2 * (
                                                               1 - pred_next_h1), pred_next_h1 * pred_next_h2],
                                                  axis=-1) + 1e-9)

        print('loss 2')
        reconstruction = tf.reduce_mean(tf.reduce_sum(-genotypes * reconstruction_p1, axis=[1, 2]), -1)
        reconstruction_next = tf.reduce_mean(tf.reduce_sum(-next_step_genotypes * reconstruction_next_p1, axis=[1, 2]),
                                             -1)
        rec_obj = reconstruction + lmbda_next * reconstruction_next

        print('loss 3')

        if self.kl:
            if test:
                param_prior = self.prior(reuse=True)
                # this is just the KL of a discrete distribution, really easy to compute, just need to make dims of prior and posterior equal
                param_prior = tf.tile(tf.reshape(param_prior, [1, 1, -1]), [self.batch_size, self.max_hap_len, 1])
                kl_h1 = tf.reduce_mean(tf.reduce_sum(param_h1 * (tf.log(param_h1) - tf.log(param_prior)), [-2, -1]))
                kl_h2 = tf.reduce_mean(tf.reduce_sum(param_h2 * (tf.log(param_h2) - tf.log(param_prior)), [-2, -1]))
                full_kl = kl_h1 + kl_h2
                self.sum_kl_test = tf.summary.scalar(tensor=full_kl, name='kl_test')
                obj = rec_obj + full_kl
                self.sum_elbo_test = tf.summary.scalar(tensor=obj, name='train_elbo_test')


            else:
                # KL retry
                log_prob_h1 = (self.len_ancs - 1) * tf.log(self.temperature1) + tf.reduce_sum(
                    tf.log(param_h1) - self.temperature1 * lat_h1, -1) - self.len_ancs * (
                                  tf.log(tf.reduce_sum(tf.exp(tf.log(param_h1) - self.temperature1 * lat_h1), -1)))
                log_prob_h2 = (self.len_ancs - 1) * tf.log(self.temperature2) + tf.reduce_sum(
                    tf.log(param_h2) - self.temperature2 * lat_h2, -1) - self.len_ancs * (
                                  tf.log(tf.reduce_sum(tf.exp(tf.log(param_h2) - self.temperature2 * lat_h2), -1)))
                log_prob_pr_h1 = (self.len_ancs - 1) * tf.log(self.temperature_prior) + tf.reduce_sum(
                    tf.log(param_prior) - self.temperature_prior * lat_h1, -1) - self.len_ancs * (tf.log(
                    tf.reduce_sum(tf.exp(tf.log(param_prior) - self.temperature_prior * lat_h1), -1)))
                log_prob_pr_h2 = (self.len_ancs - 1) * tf.log(self.temperature_prior) + tf.reduce_sum(
                    tf.log(param_prior) - self.temperature_prior * lat_h2, -1) - self.len_ancs * (tf.log(
                    tf.reduce_sum(tf.exp(tf.log(param_prior) - self.temperature_prior * lat_h2), -1)))
                full_kl = tf.reduce_mean(
                    tf.reduce_sum((log_prob_h1 - log_prob_pr_h1 + log_prob_h2 - log_prob_pr_h2), -1), -1)
                self.sum_kl_train = tf.summary.scalar(tensor=full_kl, name='kl_train')
                obj = rec_obj + full_kl
                self.sum_elbo_train = tf.summary.scalar(tensor=obj, name='train_elbo')



        else:
            obj = rec_obj
            full_kl = tf.constant(0, dtype=tf.float32)
            log_prob_h1 = tf.constant(0, dtype=tf.float32)
            log_prob_pr_h1 = tf.constant(0, dtype=tf.float32)
            if test:
                self.sum_kl_test = tf.summary.scalar(tensor=full_kl, name='kl_test')
                self.sum_elbo_test = tf.summary.scalar(tensor=obj, name='train_elbo_test')
            else:
                self.sum_kl_train = tf.summary.scalar(tensor=full_kl, name='kl_train')
                self.sum_elbo_train = tf.summary.scalar(tensor=obj, name='train_elbo')

        if test:
            self.sum_rec_test = tf.summary.scalar(tensor=reconstruction, name='reconstruction_test')
            self.sum_rec_next_test = tf.summary.scalar(tensor=lmbda_next * reconstruction_next,
                                                       name='reconstruction_next_test')

        else:
            self.sum_rec_train = tf.summary.scalar(tensor=reconstruction, name='reconstruction_train')
            self.sum_rec_next_train = tf.summary.scalar(tensor=lmbda_next * reconstruction_next,
                                                        name='reconstruction_next_train')

        if ort_constr:
            if constr_q:
                ort = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(q_h1, [self.batch_size * self.max_hap_len, self.len_ancs]) * (
                        tf.reshape(q_h2, [self.batch_size * self.max_hap_len, self.len_ancs])), -1), -1)
            else:
                num = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs])), -1), -1)
                denom = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs])), -1),
                                       -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs])), -1), -1)
                ort = tf.divide(num, denom)

                num_d = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                       -1)
                denom_d = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                         -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                                              -1)
                ort_d = tf.divide(num_d, denom_d)

                num_d2 = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1), -1)
                denom_d2 = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1),
                                          -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1), -1)
                ort_d2 = tf.divide(num_d2, denom_d2)

                ort = ort + ort_d + ort_d2

            obj = obj + ort
            self.ort_sum = tf.summary.scalar(tensor=ort, name='orthog_constraint')
        return obj, reconstruction, full_kl

    def run_enc(self, train, units_firstlstm, batch_a_haps, hap_lens, reuse):
        print('ENC_i {}'.format(batch_a_haps))
        input_encoder = self.p1_encoder_rnn(input_encoder=batch_a_haps, reuse=reuse, hap_lens=hap_lens,
                                            units_bilstm=units_firstlstm)
        print('ENC_i2 {}'.format(input_encoder))
        input_encoder_h1, input_encoder_h2 = tf.split(input_encoder, 2, axis=-1)
        out_h1, dist_params_h1, q_h1 = self.p2_encoder2_rnn(hap_lens=hap_lens, train=train,
                                                            input_encoder=input_encoder_h1,
                                                            temperature=self.temperature1, units_lstm=units_firstlstm,
                                                            reuse=reuse)
        print('ENC_o {}'.format(out_h1))
        out_h2, dist_params_h2, q_h2 = self.p2_encoder2_rnn(hap_lens=hap_lens, train=train,
                                                            input_encoder=input_encoder_h2,
                                                            temperature=self.temperature2, units_lstm=units_firstlstm,
                                                            reuse=True)
        return out_h1, out_h2, dist_params_h1, dist_params_h2, q_h1, q_h2

    def vanilla_decoder(self, input, dense_units, reuse, hap_lens, units_bilstm, units_bilstm2):
        print('DECODER_I {}'.format(input))
        with tf.variable_scope('dec_p1', reuse=reuse):
            input = tf.layers.dense(inputs=input, units=dense_units, activation=tf.nn.relu)
            cell1 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            cell2 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            values1, states = tf.nn.bidirectional_dynamic_rnn(inputs=input, dtype=tf.float32, cell_bw=cell1,
                                                              cell_fw=cell2,
                                                              sequence_length=tf.cast(hap_lens, tf.int32))
        values1 = tf.concat(values1, 2)
        with tf.variable_scope('dec_p2', reuse=reuse):
            cell = tf.contrib.rnn.LSTMCell(num_units=units_bilstm2)
            values2, states = tf.nn.dynamic_rnn(inputs=values1, dtype=tf.float32, cell=cell)
            predictions = tf.layers.dense(inputs=values2, activation=tf.nn.sigmoid, units=1)
        print('DECODER_O {}'.format(predictions))
        return predictions, values1, values2

    def peaking_decoder(self, input):
        return

    def run_decoder(self, inp_h1, inp_h2, hap_lens, dense_units, units_bilstm, units_bilstm2, reuse):
        pred_h1, intmdt_h1, intmdt2_h1 = self.vanilla_decoder(hap_lens=hap_lens, input=inp_h1, dense_units=dense_units,
                                                              units_bilstm=units_bilstm, reuse=reuse,
                                                              units_bilstm2=units_bilstm2)
        pred_h2, intmdt_h2, intmdt2_h2 = self.vanilla_decoder(hap_lens=hap_lens, input=inp_h2, dense_units=dense_units,
                                                              units_bilstm=units_bilstm, reuse=True,
                                                              units_bilstm2=units_bilstm2)
        return pred_h1, pred_h2, intmdt_h1, intmdt_h2, intmdt2_h1, intmdt2_h2

    def run_network(self, batch_a_haps, hap_lens, ort_constr):
        if self.alt_rnn:
            X_sampled_h1, X_sampled_h2, reconstruction_h1, pred_next_rec_h1, reconstruction_h2, pred_next_rec_h2 = self.alt_encoder2_rnn(
                dense_intmdt_pred_units=self.dense_intmdt_pred_units, input_encoder=batch_a_haps, temperature=self.temperature1,
                units_lstm=self.units_firstlstm, train=True, hap_lens=hap_lens, reuse=None)
            obj, reconstruction, full_kl = self.compute_alt_loss(ort_constr=ort_constr, test=False,
                                                                 lmbda_next=self.lmbda_next, genotypes=batch_a_haps,
                                                                 lat_h1=X_sampled_h1, lat_h2=X_sampled_h2,
                                                                 pred_rec_h1=reconstruction_h1,
                                                                 pred_rec_h2=reconstruction_h2,
                                                                 pred_next_h1=pred_next_rec_h1,
                                                                 pred_next_h2=pred_next_rec_h2)
            lat_h1 = 0
            log_prob_h1 = 0
            log_prob_pr_h1 = 0

        else:
            lat_h1, lat_h2, param_h1, param_h2, q_h1, q_h2 = self.run_enc(train=True,
                                                                          units_firstlstm=self.units_firstlstm,
                                                                          batch_a_haps=batch_a_haps, hap_lens=hap_lens,
                                                                          reuse=None)
            sample_h1 = tf.exp(lat_h1)
            sample_h2 = tf.exp(lat_h2)
            reconstruction_h1, reconstruction_h2, intmdt_h1, intmdt_h2, intmdt2_h1, intmdt2_h2 = self.run_decoder(
                hap_lens=hap_lens, reuse=None, inp_h1=sample_h1, inp_h2=sample_h2, dense_units=self.dense_units,
                units_bilstm=self.units_bilstm, units_bilstm2=self.units_bilstm2)
            param_prior = self.prior(reuse=None)
            obj, reconstruction, full_kl, log_prob_h1, log_prob_pr_h1, lat_h1 = self.compute_loss(
                intmdt_h1_dec2=intmdt2_h1, intmdt_h2_dec2=intmdt2_h2, intmdt_h1_dec=intmdt_h1, intmdt_h2_dec=intmdt_h2,
                q_h1=q_h1, q_h2=q_h2, ort_constr=ort_constr, input=batch_a_haps, pred_h1=pred_h1, pred_h2=pred_h2,
                lat_h1=lat_h1, lat_h2=lat_h2, param_h1=param_h1, param_h2=param_h2, param_prior=param_prior)

        return obj, reconstruction, full_kl, log_prob_h1, log_prob_pr_h1, lat_h1, reconstruction_h1, reconstruction_h2

    def test_alt_network(self, batch_a_haps, hap_lens, ort_constr):
        X_sampled_h1, X_sampled_h2, reconstruction_h1, pred_next_rec_h1, reconstruction_h2, pred_next_rec_h2 = self.alt_encoder2_rnn(
            dense_intmdt_pred_units=self.dense_intmdt_pred_units, input_encoder=batch_a_haps, temperature=self.temperature1,
            units_lstm=self.units_firstlstm, train=False, hap_lens=hap_lens, reuse=True)
        obj, reconstruction, full_kl = self.compute_alt_loss(ort_constr=ort_constr, test=True,
                                                             lmbda_next=self.lmbda_next,
                                                             genotypes=batch_a_haps, lat_h1=X_sampled_h1,
                                                             lat_h2=X_sampled_h2, pred_rec_h1=reconstruction_h1,
                                                             pred_rec_h2=reconstruction_h2,
                                                             pred_next_h1=pred_next_rec_h1,
                                                             pred_next_h2=pred_next_rec_h2)

        return obj, reconstruction, full_kl, reconstruction_h1, reconstruction_h2

    def prior(self, reuse):
        with tf.variable_scope('prior_scope', reuse=reuse):
            prior = tf.get_variable(shape=self.len_ancs, dtype=tf.float32, name='prior_dist')
        prior = tf.nn.softmax(prior)
        return prior

    def test_network(self, batch_a_haps, hap_lens):
        lat_h1, lat_h2, param_h1, param_h2, q_h1, q_h2 = self.run_enc(hap_lens=hap_lens, train=False,
                                                                      units_firstlstm=self.units_firstlstm,
                                                                      batch_a_haps=batch_a_haps, reuse=True)
        pred_h1, pred_h2, _, _, _, _ = self.run_decoder(hap_lens=hap_lens, reuse=True, inp_h1=lat_h1, inp_h2=lat_h2,
                                                        dense_units=self.dense_units, units_bilstm=self.units_bilstm,
                                                        units_bilstm2=self.units_bilstm2)
        return pred_h1, pred_h2, lat_h1, lat_h2, param_h1, param_h2

    def compute_test_loss(self, pred_h1, pred_h2, param_h1, param_h2, input):
        param_prior = self.prior(reuse=True)
        # this is just the KL of a discrete distribution, really easy to compute, just need to make dims of prior and posterior equal
        param_prior = tf.tile(tf.reshape(param_prior, [1, 1, -1]), [self.batch_size, self.max_hap_len, 1])
        kl_h1 = tf.reduce_mean(tf.reduce_sum(param_h1 * (tf.log(param_h1) - tf.log(param_prior)), [-2, -1]))
        kl_h2 = tf.reduce_mean(tf.reduce_sum(param_h2 * (tf.log(param_h2) - tf.log(param_prior)), [-2, -1]))
        full_kl = kl_h1 + kl_h2

        reconstruction_p1 = tf.log(tf.concat(
            [(1 - pred_h1) * (1 - pred_h2), (1 - pred_h2) * pred_h1 + pred_h2 * (1 - pred_h1), pred_h1 * pred_h2],
            axis=-1) + 1e-9)

        reconstruction = tf.reduce_mean(tf.reduce_sum(-input * reconstruction_p1, axis=[1, 2]), -1)
        if self.kl:
            elbo = reconstruction + full_kl
        else:
            elbo = reconstruction

        self.sum_kl_test = tf.summary.scalar(tensor=full_kl, name='kl_test')
        self.sum_rec_test = tf.summary.scalar(tensor=reconstruction, name='reconstruction_test')
        self.sum_elbo_test = tf.summary.scalar(tensor=elbo, name='test_elbo')
        return elbo, kl_h1, kl_h2

    def compute_loss(self, intmdt_h1_dec2, intmdt_h2_dec2, intmdt_h1_dec, intmdt_h2_dec, q_h1, q_h2, pred_h1, pred_h2,
                     lat_h1, lat_h2, param_h1, param_h2, param_prior, input, ort_constr, constr_q=False):
        #
        print('Computing loss')
        print(pred_h1)
        reconstruction_p1 = tf.log(tf.concat(
            [(1 - pred_h1) * (1 - pred_h2), (1 - pred_h2) * pred_h1 + pred_h2 * (1 - pred_h1), pred_h1 * pred_h2],
            axis=-1) + 1e-9)

        print('loss 2')
        reconstruction = tf.reduce_mean(tf.reduce_sum(-input * reconstruction_p1, axis=[1, 2]), -1)
        print('loss 3')
        print('rec_o1 {}'.format(reconstruction))

        print('lat_h1 {}'.format(lat_h1))
        print('param_h1 {}'.format(param_h1))

        if self.kl:
            # KL retry
            log_prob_h1 = (self.len_ancs - 1) * tf.log(self.temperature1) + tf.reduce_sum(
                tf.log(param_h1) - self.temperature1 * lat_h1, -1) - self.len_ancs * (
                              tf.log(tf.reduce_sum(tf.exp(tf.log(param_h1) - self.temperature1 * lat_h1), -1)))
            log_prob_h2 = (self.len_ancs - 1) * tf.log(self.temperature2) + tf.reduce_sum(
                tf.log(param_h2) - self.temperature2 * lat_h2, -1) - self.len_ancs * (
                              tf.log(tf.reduce_sum(tf.exp(tf.log(param_h2) - self.temperature2 * lat_h2), -1)))
            log_prob_pr_h1 = (self.len_ancs - 1) * tf.log(self.temperature_prior) + tf.reduce_sum(
                tf.log(param_prior) - self.temperature_prior * lat_h1, -1) - self.len_ancs * (tf.log(
                tf.reduce_sum(tf.exp(tf.log(param_prior) - self.temperature_prior * lat_h1), -1)))
            log_prob_pr_h2 = (self.len_ancs - 1) * tf.log(self.temperature_prior) + tf.reduce_sum(
                tf.log(param_prior) - self.temperature_prior * lat_h2, -1) - self.len_ancs * (tf.log(
                tf.reduce_sum(tf.exp(tf.log(param_prior) - self.temperature_prior * lat_h2), -1)))
            full_kl = tf.reduce_mean(tf.reduce_sum((log_prob_h1 - log_prob_pr_h1 + log_prob_h2 - log_prob_pr_h2), -1),
                                     -1)

            obj = reconstruction + full_kl

        else:
            obj = reconstruction
            full_kl = tf.constant(0, dtype=tf.float32)
            log_prob_h1 = tf.constant(0, dtype=tf.float32)
            log_prob_pr_h1 = tf.constant(0, dtype=tf.float32)
        self.sum_kl_train = tf.summary.scalar(tensor=full_kl, name='kl_train')
        self.sum_rec_train = tf.summary.scalar(tensor=reconstruction, name='reconstruction_train')

        self.sum_elbo_train = tf.summary.scalar(tensor=obj, name='train_elbo')
        if ort_constr:
            if constr_q:
                ort = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(q_h1, [self.batch_size * self.max_hap_len, self.len_ancs]) * (
                        tf.reshape(q_h2, [self.batch_size * self.max_hap_len, self.len_ancs])), -1), -1)
            else:
                num = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs])), -1), -1)
                denom = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h1, [self.max_hap_len * self.batch_size, self.len_ancs])), -1),
                                       -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs]) * (
                        tf.reshape(lat_h2, [self.max_hap_len * self.batch_size, self.len_ancs])), -1), -1)
                ort = tf.divide(num, denom)

                num_d = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                       -1)
                denom_d = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h1_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                         -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2]) * (
                        tf.reshape(intmdt_h2_dec, [self.max_hap_len * self.batch_size, self.units_bilstm * 2])), -1),
                                                              -1)
                ort_d = tf.divide(num_d, denom_d)

                num_d2 = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1), -1)
                denom_d2 = tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h1_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1),
                                          -1) * tf.reduce_mean(tf.reduce_sum(
                    tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm]) * (
                        tf.reshape(intmdt_h2_dec2, [self.max_hap_len * self.batch_size, self.units_bilstm])), -1), -1)
                ort_d2 = tf.divide(num_d2, denom_d2)

                ort = ort + ort_d + ort_d2

            obj = obj + ort
            self.ort_sum = tf.summary.scalar(tensor=ort, name='orthog_constraint')
        return obj, reconstruction, full_kl, log_prob_h1, log_prob_pr_h1, lat_h1
        # , kl_p1_h1, kl_p2_h1,kl_h1,kl_h2,pred_h1,lat_h1,param_prior,kl_p1_h1_p1,kl_p1_h1_p2,kl_p1_h1_p2_2,kl_p1_h1_p2_3,kl_p2_h1_log


if __name__ == '__main__':
    # extra idea for decoder, show it the genotypes as well!
    # Fake data
    #    max_hap_len = 130
    #    haps = np.random.randint(low=0,high=2,size=[100,130])
    #    classes = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    #    haps_mat = np.zeros([100,130,3])
    #    for k,i in enumerate(haps):
    #        for l,j in enumerate(i):
    #            haps_mat[k,l,j]=1

    # load data
    data = np.load('data/hap_mat_data.npz')

    haps_mat = data['train']
    batch_a_haps = data['train']
    max_hap_len = data['max_hap_len']

    hap_lens = np.random.randint(low=0, high=130, size=100, dtype=np.int32)

    encoder = encoders(max_hap_len=max_hap_len, units_bilstm2=10, units_bilstm=10, dense_units=10, units_firstlstm=10,
                       haps=haps_mat, hap_lens=hap_lens, len_ancs=10, temperature=0.1, batch_size=100, dim_ancs=40,
                       temperature_prior=0.5)
    train_cost, reconstruction, full_kl = encoder.run_network(batch_a_haps=batch_a_haps)
    pred_h1, pred_h2, lat_h1, lat_h2, param_h1, param_h2 = encoder.test_network(batch_a_haps=batch_a_haps)

    test_cost = encoder.compute_test_loss(pred_h1=pred_h1, pred_h2=pred_h2, param_h1=param_h1, param_h2=pred_h2,
                                          input=batch_a_haps)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(
            [train_cost, test_cost, pred_h1, pred_h2, lat_h1, lat_h2, param_h1, param_h2, reconstruction, full_kl])


