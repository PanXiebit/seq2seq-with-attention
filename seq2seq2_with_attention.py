import tensorflow as tf

class seq2seq(tf.keras.Model):
    def __init__(self, input, target, vocab_size_input, vocab_size_target, embeding_dim,
                 hidden_units, batch_size):
        super(seq2seq, self).__init__()
        self.input_sent = input
        self.target_sent = target
        self.vocab_size_input = vocab_size_input
        self.vocab_size_target = vocab_size_target
        self.embedding_dim = embeding_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size

        enc_out, enc_state = self._Encoder(self.input_sent)
        self.logits = self._Decoder(enc_out, enc_state, self.target_sent)
        self.loss = self.loss(self.target_sent, self.logits)

    def _Encoder(self, input):
        """

        :param input: [batch_size, max_length_input]
        :return:
        """
        self.enc_embedding = tf.keras.layers.Embedding(self.vocab_size_input, self.embedding_dim)
        self.enc_gru = self.rnn(self.hidden_units)

        input = self.enc_embedding(input)        # [batch, max_length_input, embedding_dim]
        output, state = self.enc_gru(input, initial_state=tf.zeros((self.batch_size, self.hidden_units)))
        return output, state  # [batch_size, max_length_input, hidden_units], [batch_size, hidden_units]

    def _Decoder(self, enc_output, enc_state, target):
        """

        :param enc_output: [batch_size, max_length_input, hidden_units]
        :param target: [batch_size, max_lenght_target]
        :param enc_state: [batch_size, hidden_units]
        :return:
        """
        # decoder input
        self.dec_embedding = tf.keras.layers.Embedding(self.vocab_size_target, self.embedding_dim)
        self.target = self.dec_embedding(target[:, :-1])   # [batch_size, max_lenght_target-1, embedding_dim] 去掉最后的 <end>

        # decoder gru hidden
        self.dec_gru = self.rnn(self.hidden_units)
        dec_output, _ = self.dec_gru(inputs=self.target,
                                     initial_state=enc_state) # [batch_size, max_length_target - 1, hidden_units]

        # attention: score = fc(tanh(W1*encoder_out + W2*decoder_out))
        self.W1 = tf.keras.layers.Dense(self.hidden_units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(self.hidden_units, use_bias=False)
        self.V = tf.keras.layers.Dense(1)

        contexts_vector = []
        for i in range(dec_output.shape[1]):
            # score
            score = tf.nn.tanh(self.W1(enc_output) + self.W2(dec_output[:,i:i+1,:])) # [batch_size, max_length_input, hidden_units]
            score = self.V(score) # [batch_size, max_length_input, 1]
            # context vector
            attention_weights = tf.nn.softmax(score, axis=1)  # probality # [batch_size, max_length_input, 1]
            context_vector = attention_weights * enc_output   # [batch_size, max_length_input, hidden_units]
            context_vector = tf.reduce_sum(context_vector, axis=1, keep_dims=True) # [batch_size, 1, hidden_units]
            contexts_vector.append(context_vector) # list, the length is (max_length_target -1)
        contexts_vector = tf.concat(contexts_vector, axis=1) # all the context_vector [batch_size, max_length_target-1, hidden_units]

        # attention vector
        attention_vector = tf.nn.tanh(tf.concat([contexts_vector, dec_output], axis=-1)) # [batch_size, max_length_target-1, hidden_units*2]

        # prediction
        self.pred = tf.keras.layers.Dense(self.vocab_size_target)
        logits = self.pred(attention_vector)   # [batch_size, max_length_target -1, vocab_size_target]

        return logits

    @staticmethod
    def rnn(units):
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU(units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')

        else:
            return tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       recurrent_activation='sigmoid')

    def loss(self,target, logits):
        """
        :param target:  [batch, max_lenght_target]
        :param logits: the output of decoder [batch, max_length_target -1, vocab_size, target]
        :return:
        """
        labels = target[:, 1:]  # [batch, max_length_target - 1] 取掉 <start>
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        labels = tf.one_hot(labels, depth=self.vocab_size_target)  # [batch_size, max_length_target -1, vocab_size_target]
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)   # [batch_size, max_length_target -1]
        loss = tf.reduce_sum(loss * mask) / self.batch_size  # 这里处以batch_size, 相当于求得每句 sentence 的loss
        return loss



