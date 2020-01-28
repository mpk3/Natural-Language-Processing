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


# W/O Padding
# r_s = tf.ragged.constant(encoded_sentences)
# r_l = tf.ragged.constant(encoded_labels)

# dataset = tf.data.Dataset.from_tensors((r_s, r_l))
# dataset = dataset.padded_batch(100)

# Padding
train_padded_sentences = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_sentences[0:13000],
                           padding='post',
                           maxlen=100)
train_padded_labels = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_labels[0:13000],
                           padding='post',
                           maxlen=100)

test_padded_sentences = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_sentences[13000:],
                           padding='post',
                           maxlen=100)
test_padded_labels = tf.keras.preprocessing.\
    sequence.pad_sequences(encoded_labels[13000:],
                           padding='post',
                           maxlen=100)

train_data = tf.data.Dataset.from_tensors((train_padded_sentences,
                                           train_padded_labels))
test_data = tf.data.Dataset.from_tensors((test_padded_sentences,
                                          test_padded_labels))

train_batches = train_data
test_batches = test_data

model = tf.keras.Sequential([
  layers.Embedding(encoder.vocab_size, 16),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(100, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=1,
    validation_data=test_batches, validation_steps=1)










