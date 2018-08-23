from data_utils import load_dataset, batch_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from seq2seq2_with_attention import seq2seq
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
print(tfe.executing_eagerly())

# prepare for dataset
num_examples = 3000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(num_examples=num_examples)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)


vocab_size_input = len(inp_lang.word2idx)   # 1899
vocab_size_target = len(targ_lang.word2idx) # 919
batch_size = 60
num_epochs = 10
embedding_dim = 256
hidden_units = 256
learning_rate = 0.01

# add all variable in loss function
# def wrapper_loss(input_sent, target_sent):
#     model = seq2seq(input_sent,target_sent,
#                     vocab_size_input=vocab_size_input,
#                     vocab_size_target=vocab_size_target,
#                     embeding_dim=embedding_dim,
#                     hidden_units=hidden_units,
#                     batch_size=batch_size)
#     print(len(model.variables))
#     return model.loss
#
# loss_and_grads_fn = tfe.implicit_value_and_gradients(wrapper_loss)

global_step = tfe.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# """Creates a `Dataset` whose elements are slices of the given tensors.
buffer_size = len(input_tensor)
train_data = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
train_data = train_data.batch(batch_size, drop_remainder=True)

for epoch in range(num_epochs):
    start = time.time()
    for (batch, (input_sent, target_sent)) in enumerate(train_data):
        # loss and gradients
        print(input_sent.shape, target_sent.shape)
        with tf.GradientTape() as tape:
            model = seq2seq(input_sent, target_sent,
                            vocab_size_input=vocab_size_input,
                            vocab_size_target=vocab_size_target,
                            embeding_dim=embedding_dim,
                            hidden_units=hidden_units,
                            batch_size=batch_size)
            loss = model.loss
            variables = model.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            # print(model.variables[0], global_step.numpy())

        if global_step % 100 == 0:
            print('step{} Loss {:.4f}'.format(global_step, loss.numpy()))

    print('Time taken for one epoch {} sec\n'.format(time.time() - start))
