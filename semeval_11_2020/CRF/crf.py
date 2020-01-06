import sys
import scipy.stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
# from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle
import glob
from sklearn.model_selection import train_test_split


LAB_DAT = '/home/mpk3/Natural_Language_Processing' + \
    '/semeval_11_2020/labeled_data'


def generate_baseline(y_test):
    baseline = []
    for tag_list in y_test:
        baseline.append(['o' for i in range(len(tag_list))])
    return baseline


def separate_spans(X):
    X_span = []
    for sentence in X:
        spans = [feat_map.pop('span') for feat_map in sentence]
        X_span.append(spans)
    return X_span, X


def separate_articles(X):
    X_articles = []
    for sentence in X:
        articles = [feat_map.pop('article') for feat_map in sentence]
        X_articles.append(articles)
    return X_articles, X


def one_hot_encoder(feature_key, X):
    '''DEPRECATED
    Creates one hot encoding for whatever key is passed to it'''
    j = set()
    x = [[j.add(token[feature_key]) for token in sentence]
         for sentence in X]
    del x
    one_hot = OneHotEncoder()
    j = np.array(list(j)).reshape(-1, 1)
    one_hot.fit(j)
    return one_hot


def encode(encoder_list, X):
    '''Changes X so it is one_hot encoded'''

    for sentence in X:
        for feature_map in sentence:
            for encoder in encoder_list:
                enc = encoder[0]
                key = encoder[1]
                encodings = list(enc.transform([[feature_map[key]]])
                                 .toarray()[0])
                encodingdict = {str(ind): val
                                for ind, val in enumerate(encodings)}

                feature_map[key+'_emb'] = encodingdict
        return X


def remove_feature(key, X):
    '''Used to remove features in case I want to experiment with
    with stuff'''
    for sentence in X:
        for feature_map in sentence:
            z = feature_map.pop(key)
            del z  # This doesnt need to be done
    return X


def flatten_sent_embedding(sent):
    '''CRF suite has very particular requirements for the shape of
    the data that you pass to it. This function flattens embeddings
    and turns them into dictionaries.
    
    Not a lot of embeddings were used in this because it greatly 
    increases the dimensionality of the features and presents memory
    issues when working with the whole data set
    
    '''
    # Deprecated; Will be used later
    for token_dict in sent:
        emb = token_dict.pop('embedding')
        emb_list = list(enumerate(emb))
        for i, flt in emb_list:
            token_dict[str(i)] = float(flt)  # Also converts to float
    return sent


def competition_output(f_out, y_pred, test_spans, test_articles):
    with open(f_out, 'a+') as sub:
        i = 0
        for sentence in y_pred:
            j = 0
            article = test_articles[i][j]
            for prediction in sentence:
                if prediction is 'p':
                    start = test_spans[i][j][0]
                    fin = test_spans[i][j][0]
                    line = article + '\t' + start + '\t' + fin
                    sub.write(line + '\n')
                j = j + 1
            i = i + 1


def test_function(feature_set, X):
    '''Removes all features not in feature_set from X'''
    keys = set(X[0][0].keys())
    for key in keys:
        if key not in feature_set:
            X = remove_feature(key, X)
    return X


# Test Sets
# N-Grams

unigram =\
    ['pos', 'token',  'main_is_stop', 'main_is_lower']
trigram = unigram + \
    ['prev_pos', 'next_pos'] +\
    ['prev_tok', 'next_tok'] +\
    ['prev_is_stop', 'next_is_stop'] +\
    ['prev_is_lower',  'next_is_lower']
fourgram = trigram +\
    ['prev_prev_pos', 'next_next_pos'] +\
    ['prev_prev_tok', 'next_next_tok'] +\
    ['prev_prev_is_stop', 'next_next_is_stop'] +\
    ['prev_prev_is_lower', 'next_next_is_lower']
n_gram_trials = [unigram, trigram, fourgram]


trials = n_gram_trials
TRIAL = '/trial9/'
fin = LAB_DAT + TRIAL + "*.pickle"
files = glob.glob(fin)
X = []
Y = []

for article in files:  # files_test:
    sentences = pickle.load(open(article, "rb"))
    sentences = sentences[1:]  # Removes title sentence
    for X_i in sentences:
        Y_i = [y.pop('class') for y in X_i]
        X.append(X_i)
        Y.append(Y_i)

X[0][0]

i = 0
for trial_feature_list in trials:
    test_feat_set = set(trial_feature_list)
    Z = X
    Z = test_function(test_feat_set, Z)

    X_train, X_test, y_train, y_test = train_test_split(Z,
                                                        Y,
                                                        test_size=0.20,
                                                        random_state=42)
    crf = sklearn_crfsuite.CRF(
        algorithm='arow',
        max_iterations=100,
        all_possible_states=True,
        all_possible_transitions=True,
        verbose=True)

    set_f_out = str(i) + '_features.txt'
    sys.stdout = open(set_f_out, "w")

    print (Z[0][0].keys())
    sys.stdout.close()
    iterations = str(i) + '_iterations.txt'
    sys.stdout = open(iterations, "w")
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    sys.stdout.close()

    results = str(i) + '_results.txt'
    sys.stdout = open(results, "w")
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)
    print(metrics.flat_classification_report(y_test, y_pred))
    sys.stdout.close()
    i = i + 1



defx = 0



















# Trials
TRIAL = '/trial9/'
fin = LAB_DAT + TRIAL + "*.pickle"

files = glob.glob(fin)
amount = files  # files[]
X = []
Y = []

# files_test = files[0:3]
for article in amount:  # files_test:
    sentences = pickle.load(open(article, "rb"))
    sentences = sentences[1:]  # Removes title
    for X_i in sentences:
        Y_i = [y.pop('class') for y in X_i]
        #X_i = flatten_sent_embedding(X_i)
        X.append(X_i)
        Y.append(Y_i)
        

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.20,
                                                    random_state=42)

# Necessary; This information also is necessary for later scoring
train_spans, X_train = separate_spans(X_train)
test_spans, X_test = separate_spans(X_test)
train_articles, X_train = separate_articles(X_train)
test_articles, X_test = separate_articles(X_test)


# Flair Sentiment Information Removal
# Generally speaking sentiment was not important for this task so far
# I used two separate models but generally didnt see any significant
# performance increase. NLTK sentiment was left in the model
X_train = remove_feature('sentence_pos_or_neg', X_train)
X_train = remove_feature('sentence_sentiment_score', X_train)
X_test = remove_feature('sentence_pos_or_neg', X_test)
X_test = remove_feature('sentence_sentiment_score', X_test)
X_train = remove_feature('token_pos_or_neg', X_train)
X_train = remove_feature('token_sentiment_score', X_train)
X_test = remove_feature('token_pos_or_neg', X_test)
X_test = remove_feature('token_sentiment_score', X_test)
# X_train = remove_feature('in_title', X_train)
# X_test = remove_feature('in_title', X_test)

X_train = remove_feature('vader_sentence', X_train)
X_test = remove_feature('vader_sentence', X_test)
X_train = remove_feature('vader_token', X_train)
X_test = remove_feature('vader_token', X_test)
# X_train = remove_feature('token', X_train)
# X_test = remove_feature('token', X_test)


crf = sklearn_crfsuite.CRF(
    algorithm='ap',
    max_iterations=100,
    all_possible_states=True,
    all_possible_transitions=True,
    verbose=True)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

baseline = generate_baseline(y_pred)
print(metrics.flat_classification_report(y_test, y_pred))
print(metrics.flat_classification_report(y_test, baseline))

X_train[0][0]





# From original v1 model; not done yet for v2
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted')

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=2,
                        verbose=1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)

y_pred = rs.predict(X_test)

metrics.flat_f1_score(y_test, y_pred, average='weighted')
metrics.flat_f1_score(y_test, baseline, average='weighted')
metrics.flat_accuracy_score(y_test, y_pred)
metrics.flat_recall_score(y_test, y_pred)
print(metrics.flat_classification_report(y_test, y_pred))
print(metrics.flat_classification_report(y_test, baseline))


#
#
#
#  Optimization:


crf = rs.best_estimator_
y_pred = crf.predict(X_test)


# Encoding Tests

'''
pos_one_hot = one_hot_encoder('pos', X)
sentiment_one_hot = one_hot_encoder('token_pos_or_neg', X)
token_one_hot = one_hot_encoder('token', X)
'''

##### Encoder Test Stuff
'''
keys = set(X[0][0].keys())
keys.remove('article')
keys.remove('span')
keys.remove('next_tok')
keys.remove('prev_tok')
#keys.remove('token')
keys.remove('sentence_pos_or_neg')
keys.remove('sentence_sentiment_score')
#keys.remove('token_pos_or_neg')
keys.remove('token_sentiment_score')
keys.remove('prev_pos')
keys.remove('next_pos')
#keys.remove('pos')
encoders = [(one_hot_encoder(key, X), key) for key in keys]
X_train = encode(encoders, X_train)
X = encode(encoders, X)
# X = encode() '''
