from flair.models import TextClassifier
from flair.data import Sentence
import gc
import glob
import pickle
# import flair

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
        self.article_name = ''
        self.main_embed = []
        self.prev_embed = []
        self.next_embed = []

    def pickle_dump(self, obj, fout):
        pickle.dump(obj, open(fout, "wb"))

    def clear(self):
        ''' CLEAR !'''

        self.sentences = []
        self.tokenized_sentences = []
        self.sentence_sentiments = []
        self.token_level_sentiment = []
        self.article_name = ''
        self.main_embed = []
        self.prev_embed = []
        self.next_embed = []

    def load_article_pickle(self, pickle_list):
        '''Loads the pickle file created for each article.
        Each row corresponds to a sentence of that article
        Each column corresponds to a dictionary of features
        for the token at that index.
        '''
        self.sentences = pickle.load(open(pickle_list, 'rb'))
        self.article_name = self.sentences[0][0]['article']

    def separate_tokens(self):
        '''Creates a separate token lists for the sentences
        in order to perform additional feature extraction'''
        for sentence in self.sentences:
            self.tokenized_sentences.append([feat_map['token']
                                             for feat_map in sentence])

    def sentence_sentiment_analysis(self):
        '''Gets the sentiment of the entire sentence:
        Score : float score
        pos_or_neg: sentiment'''
        if self.sentiment_model is None:
            self.sentiment_model = TextClassifier.load('en-sentiment')
        for token_sentence in self.tokenized_sentences:
            sep = ' '
            single_string = sep.join(token_sentence)
            sentence_obj = Sentence(single_string)
            self.sentiment_model.predict(sentence_obj)
            labels = sentence_obj.labels[0]
            score = labels.score
            pos_or_neg = labels.value
            self.sentence_sentiments.append((pos_or_neg, score))

    def token_sentiment_analysis(self):
        '''Perform sentiment analysis on the tokenized sentences on
        each token
        '''
        if self.sentiment_model is None:
            self.sentiment_model = TextClassifier.load('en-sentiment')
        for token_sentence in self.tokenized_sentences:
            sentence = []
            for token in token_sentence:
                token_obj = Sentence(token)
                self.sentiment_model.predict(token_obj)
                labels = token_obj.labels[0]
                score = labels.score
                pos_or_neg = labels.value
                sentence.append((pos_or_neg, score))
            self.token_level_sentiment.append(sentence)

    def retag_sentiment(self):
        ''' Adds the new features to the json objects in self.sentences'''
        i = 0
        for sent in self.sentences:
            j = 0
            for token_feature_map in sent:
                token_feature_map['token_pos_or_neg'] =\
                    self.token_level_sentiment[i][j][0]
                token_feature_map['token_sentiment_score'] =\
                    self.token_level_sentiment[i][j][1]
                token_feature_map['sentence_pos_or_neg'] =\
                    self.sentence_sentiments[i][0]
                token_feature_map['sentence_sentiment_score'] =\
                    self.sentence_sentiments[i][1]
                j = j + 1
            i = i + 1

# Driver Test


# Driver
retag = Retagger()
TRIAL = LAB_DATA + '/trial1/*'
NEW_TRIAL = LAB_DATA + '/trial2/'
files = glob.glob(TRIAL)
amount = files  # [files[0]]
for f_in in amount:
    retag.load_article_pickle(f_in)
    retag.separate_tokens()
    retag.sentence_sentiment_analysis()  # Holding off on this
    retag.token_sentiment_analysis()
    retag.retag_sentiment()
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    print(retag.article_name)
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()
    gc.collect()
