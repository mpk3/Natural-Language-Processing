import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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

        tokens[:0] = ['<start>']
        tokens.append('<end>')
        i_o[:0] = ['<start>']
        i_o.append('<end>')

        tokenized_sentences.append(tokens)
        labels.append(i_o)

    vocab_set.add('<start>')
    vocab_set.add('<end>')
    label_set.add('<start>')
    label_set.add('<end>')

    return tokenized_sentences, labels, vocab_set, label_set


def pad_encode(vocab_set, tokenized_sentences, label_set, labels):
    '''Encodes and pads sentences and labels'''
    encoder = tfds.features.text.TokenTextEncoder(vocab_set)
    label_encoder = tfds.features.text.TokenTextEncoder(label_set)

    # Bottleneck: Encoding. This could be done better
    encoded_sentences = [encoder.encode(' '.join(sent))
                         for sent in tokenized_sentences]
    encoded_labels = [label_encoder.encode(' '.join(sent))
                      for sent in labels]

    padded_sentences = tf.keras.preprocessing.\
        sequence.pad_sequences(encoded_sentences,
                               padding='post',
                               maxlen=100)
    padded_labels = tf.keras.preprocessing.\
        sequence.pad_sequences(encoded_labels,
                               padding='post',
                               maxlen=100)

    return padded_sentences, padded_labels

# Loading Data
data = pickle.load(open(DATA_DIR, 'rb'))
tok_sent, labels, vocab_set, label_set = preprocess_data(data)

input_tensor, target_tensor = pad_encode(vocab_set, tok_sent,
                                         label_set, labels)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val =\
    train_test_split(input_tensor, target_tensor, test_size=0.2)


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(vocab_set)+1
vocab_tar_size = len(label_set)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)






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










