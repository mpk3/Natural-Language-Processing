import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
from tensorflow.keras import layers

DATA_DIR = '/home/mpk3/Natural_Language_Processing/' +\
    'semeval_11_2020/nn/data/clean_data.pickle'


def preprocess_data(d):
    vocab_set = set()
    label_set = set('o' 'p')
    tokenized_sentences = []
    labels = []
    for sentence in data:
        tokens = []
        i_o = []
        for token in sentence:
            tokens.append(token[0])
            i_o.append(token[1])
            vocab_set.add(token[0])
        tokenized_sentences.append(tokens)
        labels.append(i_o)
    return tokenized_sentences, labels, vocab_set, label_set


def encode(vocab_set, tokenized_sentences, label_set, labels):
    '''Encodes tokenized sentence matrix into integers and then
    pads them to a length of 100
    This also originally padded sentences with keras api but I am
    switching over to using tensorflows Dataset class to get more
    familiar with the built in preprocessing classes/methods in TF'''
    encoder = tfds.features.text.TokenTextEncoder(vocab_set)
    label_encoder = tfds.features.text.TokenTextEncoder(label_set)

    # Bottleneck: Encoding. This could be done better
    encoded_sentences = [encoder.encode(' '.join(sent))
                         for sent in tokenized_sentences]
    encoded_labels = [label_encoder.encode(' '.join(sent))
                      for sent in labels]

    everything = tuple(zip(encoded_sentences, encoded_labels))
    return everything

# Loading Data
data = pickle.load(open(DATA_DIR, 'rb'))
tok_sent, labels, vocab_set, label_set = preprocess_data(data)

# Encoding
encoder = tfds.features.text.TokenTextEncoder(vocab_set)
label_encoder = tfds.features.text.TokenTextEncoder(label_set)
encoded_sentences = [encoder.encode(' '.join(sent))
                     for sent in tok_sent]
encoded_labels = [label_encoder.encode(' '.join(sent))
                  for sent in labels]

r_s = tf.ragged.constant(encoded_sentences)
r_l = tf.ragged.constant(encoded_labels)

# Padding
padded_sentences = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_sentences,
                           padding='post',
                           maxlen=100)
padded_labels = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_labels,
                           padding='post')

# W/O Padding

dataset = tf.data.Dataset.from_tensor_slices((r_s,
                                              r_l))

dataset = dataset.shuffle(5000)
dataset = dataset.padded_batch(100,
                               tf.compat.v1.data.get_output_shapes(dataset))



dataset = tf.data.Dataset.from_tensor_slices((padded_sentences,
                                              padded_labels))

dataset = dataset.batch(5000)

train = dataset.skip(2000)
test = dataset.take(2000)

train = train
test = test
embedding = layers.Embedding(input_dim=(len(vocab_set)+2),
                             output_dim=16, mask_zero=True)

masked_output = embedding(train)

model = tf.keras.Sequential()
model.add(embedding)
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



