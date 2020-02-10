import gc
from itertools import chain
from matplotlib import pyplot as plt
import pickle
import glob
import numpy as np
import pandas as pd

FILES_IN = '/home/mpk3/Natural_Language_Processing/' +\
    'semeval_11_2020/labeled_data/trial2/*'


class Cleaner():
    '''Class for cleaning and removing old formatting from CRF
    The rnn itself has something similar to this. This class however
    is also used to establish some basic metrics about the dataset'''

    def __init__(self):
        self.data = []

    def load_data(self, FILES_IN):
        '''Loading all the sentences for inspection and basic encoding.
        This also strips stuff from the earlier formats I built'''
        files = glob.glob(FILES_IN)
        for fi in files:
            sentences = pickle.load(open(fi, 'rb'))
            for sentence in sentences:
                self.data.append(sentence)
                # self.data = np.vstack((self.data, sentence))

    def clean_data(self):
        '''Removes unneccessary stuff I used for a different model'''
        for i, sentence in enumerate(self.data):
            for j, word in enumerate(sentence):
                self.data[i][j] = (word['token'], word['class'],
                                   word['span'], word['article'])
 

c = Cleaner()
c.load_data(FILES_IN)
c.clean_data()

# Find max length
lengths = [len(i) for i in c.data]
df = pd.DataFrame(lengths)
fig, ax = plt.subplots()
df.plot(ax=ax)
xs = np.linspace(1, 21, len(df))
ax.axhline(y=df.median().iloc[0], xmin=0, xmax=len(xs),
           color='r', linestyle='--', lw=2)
ax.axhline(y=df.mean().iloc[0], xmin=0, xmax=len(xs),
           color='r', linestyle='--', lw=2)
plt.show()



max_sentence_length = len(max(c.data, key=lambda i: len(i)))
cols = ['Token', 'Class', 'Span', 'Doc']
flat_data = list(chain.from_iterable(c.data))
df = pd.DataFrame(flat_data, columns=cols)

plt.show()
df.info()

num_tok = len(df)

#df['Class'].hist(bins=2, color=['r','b'])

plt.show()

pickle.dump(c.data, open('clean_data.pickle', 'wb'))
