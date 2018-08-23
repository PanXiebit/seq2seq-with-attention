import tensorflow as tf
import os
import chardet
import re
from sklearn.model_selection import train_test_split


"""
dataset provided by http://www.manythings.org/anki/.
This dataset contains language translation pairs in the format:

May I borrow this book?    ¿Puedo tomar prestado este libro?
"""

# download and prepare the dataset
dataset_path = "./"
if not os.path.exists("./datasets/spa-eng"):
    path_to_zip = tf.keras.utils.get_file(fname="spa-eng.zip",
                                          origin='http://download.tensorflow.org/data/spa-eng.zip',
                                          md5_hash=True,
                                          extract=True,
                                          cache_dir=dataset_path)
else:
    path_to_zip = dataset_path + "datasets/spa-eng.zip"
# print(path_to_zip)  # ./datasets/spa-eng.zip


# Returns the directory component of a pathname
path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
# print(path_to_file)

"""
# 查看文件的编码格式
with open(path_to_file, "rb") as f:
    data = f.read()
    print(chardet.detect(data))  # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

"""

# 查看文件内容具体形式
with open(path_to_file, "r") as f:
    for i, line in enumerate(f):
        if i > 3:
            break
        # print(line)
"""
b'Go.\tVe.\n'
b'Go.\tVete.\n'
b'Go.\tVaya.\n'
b'Go.\tV\xc3\xa1yase.\n'
b'Hi.\tHola.\n'
b'Run!\t\xc2\xa1Corre!\n'
"""

# 预处理
def preprocess_sentence(sent):
    sent = sent.lower().strip()
    sent = re.sub(r"([?.!,])", r" \1 ", sent)  # eg: "he is a boy." => "he is a boy ."
    sent = re.sub(r'[" "]+', " ", sent)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sent = re.sub(r"[^a-zA-Z?.!,]+", " ", sent)

    sent = sent.rstrip().strip()

    sent = "<start> " + sent + " <end>"
    return sent

# remove the accents
# clean the sentences
# return word pairs in the format: [English, Spanish]
def create_dataset(path, num_examples):
    word_pairs = []
    with open(path, encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i > num_examples:
                break
            line = line.strip().split("\n")
            word_pair = [preprocess_sentence(sent) for l in line for sent in l.split('\t')]
            word_pairs.append(word_pair)
    return word_pairs       # eg. [['<start> go . <end>', '<start> ve . <end>'], ['<start> go . <end>', '<start> vete . <end>']]

# print(create_dataset(path_to_file, 3))

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()   # 这句话不能忘了,在创建实例的时候会初始化

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))  # 根据语料库添加单词

        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        self.word2idx["<unk>"] = 1
        self.word2idx["<start>"] = 2
        self.word2idx['<end>'] = 3

        for index, word in enumerate(self.vocab, 4):
            if word in self.word2idx:
                continue
            self.word2idx[word] = index

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(num_examples=3000, path=path_to_file):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)

    # Vectorize the input and target languages

    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang


### Create a tf.data dataset
def batch_data(input_tensor, target_tensor, batch_size):
    buffer_size = len(input_tensor)
    # """Creates a `Dataset` whose elements are slices of the given tensors.
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
