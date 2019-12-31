from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tokenize import PunktSentenceTokenizer as pkt
from nltk import pos_tag
import glob
import pickle
import sklearn
import sklearn_crfsuite
import fasttext

MAIN = '/home/mpk3/Natural_Language_Processing/semeval_11_2020'

# Models
MODEL_DIR = MAIN + '/models'
# DATA
DATA_DIR = MAIN + '/provided/datasets'
SPAN_DIR = DATA_DIR + '/train-labels-task1-span-identification'
TECH_DIR = DATA_DIR + '/train-labels-task2-technique-classification'
ARTICLE_DIR = DATA_DIR + '/train-articles'


class Tagger():
    '''The Tagger class is responsible for all of
    the annotation/tagging needed to create features
    for the sequence model.

    The output of this class is a single pickle file
    for whatever article was passed to it. The file
    consists of N-D arrays where each row is a sentence
    and every index is a token json object whose key:value
    pairs are the features and the propaganda tag.
    The latter of course being the target value (y)'''

    def __init__(self):

        # Models
        self.token_tagger = None  # Currently twt
        self.pos_tagger = None  # Currently nltk default
        self.sent_tagger = None  # Currently punkt
        self.fasttext = None  # HUGE

        # Propaganda Span Dictionary: {'articlename':[(start, fin)]}
        self.span_dict = {}

        # Unadultered article text
        self.raw_text = ''

        # Article name
        self.article_name = ''

        # List of token positions (start, end) for each token
        self.token_spans = []

        # List of sentence positions (start, end) for each sentence

        self.sent_spans = []

        # List of lists of spans corresponding to sentences
        self.unflat_sent = []

        # List of actual tokenized sentences
        self.tokens = []

        # POS tags for correspoding tokens
        self.tags = []

        # TRI-GRAM Arrays: [(prev_tok, next_tok)]
        self.trigrams = []

        # pos trigrams
        self.pos_trigrams = []

    def clear(self):
        '''Clears all article specific variables
        Does not delete any of the models or the propaganda
        span dictionary'''

        self.raw_text = ''
        self.article_name = ''
        self.token_spans = []
        self.sent_spans = []
        self.unflat_sent = []
        self.tags = []
        self.tokens = []
        self.trigrams = []
        self.pos_trigrams = []

    def build_span_dict(self, span_directory):
        ''' This function reads in all the spans from all of the task1 directory
        The directory has a single file for each article.
        The files are tab separated and consist of the start and end positions
        of each propaganda technique. It creates a dictionary:

        self.span_dict[article_name] = [(start, end), (start, end) ...]

        There are empty files for those files without propaganda

        This can be dumped into a pickle file using pickle_dump()
        One this dictionary has been created it can be reloaded using
        load_span_dict()
        '''

        files = glob.glob(SPAN_DIR+'/*.labels')
        for article in files:
            if not os.stat(article).st_size == 0:
                spans = list(open(article, "r"))
                spans = [line.split('\t') for line in spans]
                name = spans[0][0]
                self.span_dict[name] = [(int(span[1]), int(span[2])) for span in spans]

    def pickle_dump(self, object, fout):
        pickle.dump(object, open(fout, "wb"))

    def load_span_dict(self, span_dict_pickle):
        '''This function loads an already created span dictionary
        Expects a pickle file
        '''
        self.span_dict = pickle.load(open(span_dict_pickle, 'rb'))

    def load_article(self, fin):
        '''This function reads in an article and initializes a string
        object to represent the text.

        Creates: self.raw_text
        Creates: self.article_name
        '''
        file_in = open(fin, 'r')
        doc_string = file_in.readlines()
        return

    def create_token_spans(self):
        '''This function creates a list of token spans from raw_text. The token
        indices are respective of character indices in the text.

        Creates: self.token_spans
        '''
        if self.tokenizer is None:
            self.tokenizer = twt()
        self.token_spans = list(self.tokenizer.span_tokenize(self.raw_text))

    def create_sent_spans(self):
        '''This function creates a list of sentence spans from raw_text.
        Sentence boundaries are needed because CRFs/Sequence Models require
        sequences

        Creates: self.sent_spans
        '''
        if self.sent_tagger is None:
            self.sent_tagger = pkt()
        self.sent_spans = list(self.sent_tagger.span_tokenize(self.raw_text))

    def create_sent_lists(self):
        '''Creates an array object where each row is a list of numbers
        corresponding to the tokens in each sentence. This is needed for both
        CRF/Sequence taggers and the pos_tagger which takes a list of tokens as
        its input. This function is not looking at the char level indeces of
        raw_text but instead uses the indices in token_spans

        Creates: self.unflat_sent
        '''
        sent_tok_index = [[] for sent in self.sent_spans]
        i = 0
        for st, fin in self.token_spans:
            j = 0
            for s_st, s_f in self.sent_spans:
                sent_range = set(range(s_st, s_f))
                if st in sent_range:
                    sent_tok_index[j].append(i)
                j = j + 1
            i = i + 1
        self.unflat_sent = sent_tok_index

    def create_token_list(self):
        '''Unflattened 3-D token lists.

        self.tokens = [[tokenized_sentence]]
        '''

        for sent in self.unflat_sent:
            sentence = []
            for tok_index in sent:
                sentence.append(self.token_spans[tok_index])
                token_lists.append(sentence)
            self.tokens.append(sentence)

    def pos_tag(self):
        '''Tags tokens using indices from token_spans and sent_spans
        to get strings from text_raw
        Requires: self.tokens
        Creates: self.tags
        '''
        if self.pos_tagger is None:
            self.pos_tagger = pos_tag()
        for sentence in self.tokens:
            tag_tuples = pos_tag(sentence)
            just_tags = [tup[1] for tup in tag_tuples]
            self.tags.append(just_tags)

    def generate_trigram(array):
        end = len(array) - 1
        output = []
        i = 0
        while (i < len(array)):
            if i is 0:
                behind = 'SEN_START'
            else:
                behind = array[i-1]
            if i is (len(array)-1):
                in_front = 'SEN_END'
            else:
                in_front = array[i+1]
            output.append((behind, in_front))
            i = i + 1
        return output

    def trigrams(self):
        '''Creates token trigram array
        The tuple values are tokens and not indices
        self.trigrams = [[(n-1, n+1),...][(n-1, n+1)]]
        '''
        self.trigrams = [generate_trigram(tokenized_sentence)
                         for tokenized_sentence in self.tokens]

    def pos_trigrams(self):
        '''Same as trigram but for POS tags'''
        self.pos_trigrams = [generate_trigram(pos_tag_sentence)
                             for pos_tag_sentence in self.pos_tags]
       

"""Driver"""

def create_feature_map(tagger_object):
  feat_map = {}
  feat_map[tagger_object]
   
   
        self.token_spans = []
        self.sent_spans = []
        self.unflat_sent = []
        self.pos_tags = []                                                                                                                                                                    
        self.trigrams = []

extractor = Tagger()
filter.build_span_dict(SPAN_DIR)
span_dictionary = filter.span_dict
filter.pickle_dump(span_dictionary, 'span_dictionary')

filter = Tagger()
filter.load_span_dict('span_dictionary')

files = glob.glob(ARTICLE_DIR, '/*.labels')

for file in files:
  filter.load_article(file)
  filter.create_token_spans()
  filter.create_sent_spans()
  filter.create_sent_lists() 
  filter.create_token_list()
  filter.pos_tag()
  filter.trigram()
  filter.pos_trigram()
  feature_json = filter.create_feature_json()
  filter.pickle_dump





# Tests
# Span dict tests
extractor = Tagger()
extractor.build_span_dict(SPAN_DIR)
test_array = [(1607, 1674), (143, 169), (186, 249),
              (1043, 1148), (1164, 1183), (2611, 2651)]
assert '999001970' in extractor.span_dict.keys(), 'span_dict 1'
assert test_array == extractor.span_dict['999001970'], 'span_dict 2'

# Span dict pickle tests
out_str = MODEL_DIR + '/span_dict.pickle'
extractor.pickle_dump(extractor.span_dict, out_str)
test_extractor = Tagger()
test_extractor.load_span_dict(out_str)
assert extractor.span_dict == test_extractor.span_dict, 'Load 1'
assert test_array == test_extractor.span_dict['999001970'], 'Load 2'


extractor.clear()
assert len(extractor.span_dict.keys()) is 0, 'Clear 1'
