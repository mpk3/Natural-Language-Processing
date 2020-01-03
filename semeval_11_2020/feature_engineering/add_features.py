from flair.models import TextClassifier
from flair.data import Sentence

import pickle
import flair

MAIN = '/home/mpk3/Natural_Language_Processing/semeval_11_2020'

# Models
MODEL_DIR = MAIN + '/models'

# DATA
DATA_DIR = MAIN + '/provided/datasets'
# SPAN_DIR = DATA_DIR + '/train-labels-task1-span-identification'
# TECH_DIR = DATA_DIR + '/train-labels-task2-technique-classification'
# ARTICLE_DIR = DATA_DIR + '/train-articles'

# Labeled Data
LAB_DATA = MAIN + '/labeled_data'


class Retagger:
    '''This class was created to add additional token level
    features to data already created by the tagger.py file.

    It reads in the pickled lists created by tagger.py and
    adds features to each individual token in the lists
    '''

    def __init__(self):

        # Models that dont get cleared
        self.embedding_model = None
        self.sentiment_model = None

        # Cleared
        self.sentences = []
        self.tokenized_sentences = []
        self.sentence_sentiments = []
        self.token_level_sentiment = []

    def clear(self):
        ''' CLEAR !'''
        self.sentences = []
        self.tokenized_sentences = []
        self.sentence_sentiments = []
        self.token_level_sentiment = []

    def load_article_pickle(self, pickle_list):
        '''Loads the pickle file created for each article.
        Each row corresponds to a sentence of that article
        Each column corresponds to a dictionary of features
        for the token at that index.
        '''
        self.sentences = pickle.load(open(pickle_list, 'rb'))

    def separate_tokens(self):
        '''Creates a separate token lists for the sentences
        in order to perform additional feature extraction'''
        for sentence in self.sentences:
            self.tokenized_sentences.append([feat_map['token']
                                             for feat_map in sentence])

    def sentence_sentiment(self):
        '''Gets the sentiment of the entire sentence:
        Score : float score
        pos_or_neg: sentiment'''
        if self.sentiment_model is None:
            self.sentiment_model = TextClassifier.load('en-sentiment')
        for tokenized_sentence in self.sentences:
            sep = ' '
            single_string = sep.join(tokenized_sentence)
            sentence_obj = Sentence(single_string)
            self.sentiment_model.predict(sentence_obj)
            labels = sentence_obj.labels[0]
            score = labels.score
            pos_or_neg = labels.value
            self.sentence_sentiments.append((pos_or_neg, score))

    def token_sentiment(self):
        '''Perform sentiment analysis on the tokenized sentences on
        each token
        '''
        if self.sentiment_model is None:
            self.sentiment_model = TextClassifier.load('en-sentiment')
        for tokenized_sentence in self.sentences:
            sentence = []
            for token in tokenized_sentence:
                token_obj = Sentence(token)
                self.sentiment_model.predict(sentence_obj)
                labels = token_obj.labels[0]
                score = labels.score
                pos_or_neg = labels.value
                sentence.append((pos_or_neg, score))
            self.token_level_sentiment.append(sentence)

    def retag(self):
        ''' Adds the new features to the json objects in self.sentences'''
        i = 0
# Driver Test
retag = Retagger()
retag.load_article_pickle



# Flair Test
sent_analyzer = TextClassifier.load('en-sentiment')
sep = ' '
s = ['I', '\'', 've', 'stopped', 'you']
single_string = sep.join(s)
sentence_obj = Sentence(single_string)
sent_analyzer.predict(sentence_obj)
labels = sentence_obj.labels[0]
score = labels.score
pos_or_neg = labels.value

