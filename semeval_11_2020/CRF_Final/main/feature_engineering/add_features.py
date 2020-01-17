from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import fasttext
import glob
import pickle

# FASTTEXT MODEL ORIGINALLY USED FOR EMBEDDINGS
FT_MOD = '../wiki.en.bin'

MAIN = '..'

# Models
MODEL_DIR = MAIN + '/models'

# DATA
DATA_DIR = MAIN + '/datasets'
# SPAN_DIR = DATA_DIR + '/train-labels-task1-span-identification'
# TECH_DIR = DATA_DIR + '/train-labels-task2-technique-classification'
# ARTICLE_DIR = DATA_DIR + '/train-articles'

# Labeled Data: Output Folder
LAB_DATA = MAIN + '/labeled_data'


class Retagger:
    '''This class was created to add additional token level
    features to data already created by the tagger.py file.

    It reads in the pickled lists created by tagger.py and
    adds features to each individual token in the lists
    '''

    def __init__(self):

        # Models that dont get cleared
        self.fasttext_model = None
        self.sentiment_model = None
        self.vader_model = None

        # Stopwords
        self.stopword_set = None

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

    def vader_both(self):
        '''Applies the vader model to all token features and sentence level '''
        self.separate_tokens()
        if self.vader_model is None:
            self.vader_model = SentimentIntensityAnalyzer()
        # Get Sentence Level Sentiment
        i = 0
        for token_sentence in self.tokenized_sentences:
            sep = ' '
            sentence_string = sep.join(token_sentence)
            sentence_obj = self.vader_model.polarity_scores(sentence_string)
            # Get Token level Sentiment
            for feature_map in self.sentences[i]:

                feature_map['vader_sentence'] = \
                    sentence_obj
                feature_map['vader_token'] = \
                    self.vader_model.polarity_scores(feature_map['token'])
                feature_map['vader_prev'] =\
                    self.vader_model.polarity_scores(feature_map['prev_tok'])
                feature_map['vader_next'] =\
                    self.vader_model.polarity_scores(feature_map['next_tok'])
                feature_map['vader_prev_prev'] =\
                    self.vader_model.polarity_scores(feature_map['prev_prev_tok'])
                feature_map['vader_next_next'] =\
                    self.vader_model.polarity_scores(feature_map['next_next_tok'])
            i = i + 1

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

    def retag_case_features(self):
        'Adds case related features and makes the tokens all lowercase'
        for sentence in self.sentences:
            for feature_map in sentence:

                # Main Token
                feature_map['main_is_lower'] =\
                    int(feature_map['token'].islower())
                feature_map['token'] =\
                    feature_map['token'].lower()

                # Previous
                feature_map['prev_is_lower'] =\
                    int(feature_map['prev_tok'].islower())
                feature_map['prev_tok'] =\
                    feature_map['prev_tok'].lower()

                # Next
                feature_map['next_is_lower'] =\
                    int(feature_map['next_tok'].islower())
                feature_map['next_tok'] =\
                    feature_map['next_tok'].lower()

                # Next Next Prev Prev
                if 'next_next_tok' in feature_map.keys():
                    feature_map['prev_prev_is_lower'] =\
                        int(feature_map['prev_prev_tok'].islower())
                    feature_map['prev_prev_tok'] =\
                        feature_map['prev_prev_tok'].lower()
                    feature_map['next_next_is_lower'] =\
                        int(feature_map['next_next_tok'].islower())
                    feature_map['next_next_tok'] =\
                        feature_map['next_next_tok'].lower()

    def retag_4gram(self):
        ''' Creates a 4-gram array out of sentences'''
        for sentence in self.sentences:
            i = 0
            while (i < len(sentence)):
                if i < 2:
                    two_behind = 'SEN_START'
                    pos_behind = 'SEN_START'
                else:
                    two_behind = sentence[i-2]['token']
                    pos_behind = sentence[i-2]['pos']
                if (len(sentence)-i-1) >= 2:
                    two_in_front = sentence[i+2]['token']
                    pos_in_front = sentence[i+2]['pos']
                else:
                    two_in_front = 'SEN_END'
                    pos_in_front = 'SEN_END'
                sentence[i]['prev_prev_tok'] = two_behind
                sentence[i]['next_next_tok'] = two_in_front
                sentence[i]['prev_prev_pos'] = pos_behind
                sentence[i]['next_next_pos'] = pos_in_front
                i = i + 1

    def retag_stop_words(self):
        '''Creates a stop word feature.
        Tokens should be lowercase before going through this
        '''
        if self.stopword_set is None:
            self.stopword_set = set(stopwords.words('english'))
        for sentence in self.sentences:
            for feature_map in sentence:

                # Main Token
                feature_map['main_is_stop'] =\
                    int(feature_map['token'] in self.stopword_set)

                # Previous
                feature_map['prev_is_stop'] =\
                    int(feature_map['prev_tok'] in self.stopword_set)

                # Next
                feature_map['next_is_stop'] =\
                    int(feature_map['next_tok'] in self.stopword_set)

                if 'next_next_tok' in feature_map.keys():
                    feature_map['prev_prev_is_stop'] =\
                        int(feature_map['prev_prev_tok'] in self.stopword_set)

                    feature_map['next_next_is_stop'] =\
                        int(feature_map['next_next_tok'] in self.stopword_set)

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

    def retag_title_feature(self):
        '''The first line of every file is its title. This adds a feature
        'in_title' that marks whether or not a token is in the title.
        '''
        title_set = set()
        for feature_map in self.sentences[0]:
            title_set.add(feature_map['token'])
        for sentence in self.sentences[1:]:
            for feature_map in sentence:
                if feature_map['token'] in title_set:
                    feature_map['in_title'] = 1
                else:
                    feature_map['in_title'] = 0

    def retag_token_ft_embed(self):
        '''Adds an embedding to the main token'''
        if self.fasttext_model is None:
            self.fasttext_model = fasttext.load_model(FT_MOD)
        for sentence in self.sentences:
            for feature_map in sentence:
                vec =\
                    self.fasttext_model[feature_map['token']]
                i_tuples = list(enumerate(vec))
                feature_map['token_embed'] =\
                    {str(i): flt for (i, flt) in i_tuples}

    def retag_sentiment(self):
        ''' Adds the new features to the json objects in self.sentences
        Sentence sentiment so far has actually had a negative impact on
        classification. Frankly, I think the flair model is not ideal for this
        task. I need a different kind of sentiment model or I need to finetune
        the current one.'''
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


# Driver
retag = Retagger()
TRIAL = LAB_DATA + '/trial2/*'
NEW_TRIAL = LAB_DATA + '/new/'
files = glob.glob(TRIAL)
amount = files  # [files[0]]  # files
# Vader with text case before normalization VADER FOR ALL
for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.retag_4gram()
    retag.vader_both()  # ; retag.sentences[0][0]
    retag.retag_case_features()
    retag.retag_stop_words()
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()

# First Vader Trial
'''
TRIAL = LAB_DATA + '/trial4/*'
NEW_TRIAL = LAB_DATA + '/trial7/'
files = glob.glob(TRIAL)
amount = files  # [files[0]]  # files
for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.vader_both()  # ; retag.sentences[0][0]
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()
'''
# Title Features
'''
for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.retag_title_feature()  # ; retag.sentences[0][0]
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()
'''
# 4-Gram + Fasttext
'''
4 gram information with fasttext embedding.
Didnt improve performance much and greatly increased
model complexity so working with large data sets became
problematic

for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.retag_token_ft_embed()  # ; retag.sentences[0][0]
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()
    gc.collect()
'''
# 4 gram information
'''for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.retag_4gram()  # ; retag.sentences[0][0]
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()'''

# This was for adding case and stopword information
'''for f_in in amount:
    retag.load_article_pickle(f_in)  # ; retag.sentences[0][0]
    retag.retag_case_features()  # ; retag.sentences[0][0]
    retag.retag_stop_words()  # ; retag.sentences[0][0]
    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()'''

# This version was for sentiment analysis
'''for f_in in amount:
    # retag.separate_tokens() 
    # retag.sentence_sentiment_analysis()  # Holding off on this
    # retag.token_sentiment_analysis()
    # retag.retag_sentiment()

    fout = NEW_TRIAL + retag.article_name + '_f.pickle'
    # print(retag.article_name)
    retag.pickle_dump(retag.sentences, fout)
    retag.clear()
    # gc.collect()
'''
