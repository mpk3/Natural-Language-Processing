from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tokenize import PunktSentenceTokenizer as pkt
from nltk import pos_tag
import glob
import pickle
import os

MAIN = '..'

# Models
MODEL_DIR = MAIN + '/models'

# DATA
DATA_DIR = MAIN + '/datasets'
SPAN_DIR = DATA_DIR + '/train-labels-task1-span-identification'
TECH_DIR = DATA_DIR + '/train-labels-task2-technique-classification'
ARTICLE_DIR = DATA_DIR + '/train-articles'

# Labeled Data
LAB_DATA = MAIN + '/labeled_data'


class Tagger():
    '''The Tagger class is responsible for initial phase of
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
        self.tokenizer = None  # Currently twt
        # self.pos_tagger = None  # Currently nltk default
        self.sent_tagger = None  # Currently punkt
        self.fasttext = None  # HUGE

        # Propaganda Span Dictionary: {'articlename':[(start, fin)]}
        self.span_dict = {}

        # Everything below this is cleared for each article
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

        # List of lists of tokens for each sentence
        self.tokenized_sentences = []

        # POS tags for correspoding tokens
        self.tags = []

        # TRI-GRAM Arrays: [(prev_tok, next_tok)]
        self.trigrams = []

        # pos trigrams
        self.pos_trigrams = []

        # class tags: p = propaganda, o = other
        self.class_tags = []

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
        self.tokenized_sentences = []
        self.trigrams = []
        self.pos_trigrams = []
        self.class_tags = []

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
                self.span_dict[name] = [(int(span[1]), int(span[2]))
                                        for span in spans]

    def pickle_dump(self, obj, fout):
        pickle.dump(obj, open(fout, "wb"))

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
        self.raw_text = open(fin, 'r').read()
        self.article_name =\
            fin.split('train-articles')[1].split('e')[1].split('.')[0]

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
        '''
        Used for po taggings

        Creates an array object where each row is a list of numbers
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

        self.tokenized_sentences = [[tokenized_sentence]]
        '''
        
        for span in self.sent_spans:
            sentence = []
            sent_range = set(range(span[0], span[1]))
            for token_span in self.token_spans:
                if token_span[0] in sent_range:
                    token = self.raw_text[token_span[0]:token_span[1]]
                    sentence.append(token)
            self.tokenized_sentences.append(sentence)

    def p_o_tag(self):
        '''Creates p o tagging for propaganda spans.
        p for propaganda
        o for other
        creates: self.class_tags
        This create_token_list() and create_sent_lists() could all be
        refactored.
        '''
        self.create_sent_lists()
        # Label character as other
        po_tags = ['o' for x in range(len(self.raw_text))]

        # Label everything that is propaganda as p
        if self.article_name in self.span_dict.keys():
            for prop_span in self.span_dict[self.article_name]:
                for index in range(prop_span[0], prop_span[1]):
                    po_tags[index] = 'p'

        # Create unflat class tags
        for sent_list in self.unflat_sent:
            po_tag_sent = []
            for token_index in sent_list:
                span_start = self.token_spans[token_index][0]
                po_tag_sent.append(po_tags[span_start])
            self.class_tags.append(po_tag_sent)

    def pos_tag(self):
        '''Tags tokens using indices from token_spans and sent_spans
        to get strings from text_raw
        Requires: self.tokenized_sentences
        Creates: self.tags
        '''
        for sentence in self.tokenized_sentences:
            tag_tuples = pos_tag(sentence)
            just_tags = [tup[1] for tup in tag_tuples]
            self.tags.append(just_tags)

    def generate_trigram(self, array):
        ''' Creates a trigram array out of any array passed to it'''
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

    def create_token_trigrams(self):
        '''Creates token trigram array
        The tuple values are tokens and not indices
        self.trigrams = [[(n-1, n+1),...][(n-1, n+1)]]
        '''
        self.trigrams = [self.generate_trigram(tokenized_sentence)
                         for tokenized_sentence in self.tokenized_sentences]

    def create_pos_trigrams(self):
        '''Same as trigram but for POS tags'''
        self.pos_trigrams = [self.generate_trigram(pos_tag_sentence)
                             for pos_tag_sentence in self.tags]


"""Driver"""


def create_feature_lists(fe):
    sentence_feat_list = []
    i = 0
    for sent in fe.unflat_sent:
        feature_sentence = []
        j = 0
        for token in sent:
            feature_map = {'pos': fe.tags[i][j],
                           'token': fe.tokenized_sentences[i][j],
                           'prev_tok': fe.trigrams[i][j][0],
                           'next_tok': fe.trigrams[i][j][1],
                           'prev_pos': fe.pos_trigrams[i][j][0],
                           'next_pos': fe.pos_trigrams[i][j][1],
                           'class': fe.class_tags[i][j],
                           'span': fe.token_spans[fe.unflat_sent[i][j]],
                           'article': fe.article_name}
            feature_sentence.append(feature_map)
            j = j + 1
        sentence_feat_list.append(feature_sentence)
        i = i + 1
    return sentence_feat_list


out_str = MODEL_DIR + '/span_dict.pickle'
e = Tagger()
e.load_span_dict(out_str)
files = glob.glob(ARTICLE_DIR + '/*.txt')

trial = '/trial1'
for article in files:
    try:
        e.load_article(article)
        e.create_sent_spans()
        e.create_token_spans()
        e.create_token_list()
        e.pos_tag()
        e.create_token_trigrams()
        e.create_pos_trigrams()
        e.p_o_tag()
        features = create_feature_lists(e)
        fout = LAB_DATA + trial + '/' + e.article_name + '_f.pickle'
        e.pickle_dump(features, fout)
        e.clear()
    except Exception:
        print(article)


'''
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
assert extractor.span_dict == test_extractor.span_dict, 'Load Span Dict 1'
assert test_array == test_extractor.span_dict['999001970'], 'Load Span Dict 2'

# Load article tests
article = '/home/mpk3/Natural_Language_Processing/semeval_11_2020'\
    + '/provided/datasets/train-articles/article736757214.txt'
extractor = Tagger()
extractor.load_span_dict(out_str)
extractor.load_article(article)
assert extractor.article_name == '736757214', 'Load article 1'
#assert len(extractor.raw_text) == 6254

# Tagger tests
extractor = Tagger()
extractor.load_span_dict(out_str)
extractor.load_article(article)
extractor.create_sent_spans()
extractor.create_token_spans()
extractor.create_token_list()
assert len(extractor.tokenized_sentences) ==\
    len(extractor.sent_spans), 'Sen span 1'

extractor.pos_tag()
assert len(extractor.tokenized_sentences) ==\
    len(extractor.tags), 'Pos Tag 1'

# Trigram test
test = ['0', '1', '2']
test_result = [('SEN_START', '1'), ('0', '2'), ('1', 'SEN_END')]
assert extractor.generate_trigram(test) ==\
    test_result, 'Trigram 1'
extractor.create_token_trigrams()
extractor.create_pos_trigrams()

# Class tag test
extractor.p_o_tag()
assert  len(extractor.class_tags[0]) ==\
    len(extractor.tokenized_sentences[0]), 'PO 1'

trial = '/trial1'
e = Tagger()
e.load_span_dict(out_str)
e.load_article(article)
e.create_sent_spans()
e.create_token_spans()
e.create_token_list()
e.pos_tag()
e.create_token_trigrams()
e.create_pos_trigrams()
e.p_o_tag()
features = create_feature_lists(e)
fout = LAB_DATA + trial + '/' + e.article_name + '_f.pickle'
e.pickle_dump(features, fout)
'''
