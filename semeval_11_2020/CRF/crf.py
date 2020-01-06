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

# Trials
TRIAL = '/trial6/'
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
X[0][0]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.20,
                                                    random_state=42)
train_spans, X_train = separate_spans(X_train)
test_spans, X_test = separate_spans(X_test)

train_articles, X_train = separate_articles(X_train)
test_articles, X_test = separate_articles(X_test)


# These features actually reduced accuracy. Confound with token sentiment info?
# X_train = remove_feature('sentence_pos_or_neg', X_train)
#X_train = remove_feature('sentence_sentiment_score', X_train)
#X_test = remove_feature('sentence_pos_or_neg', X_test)
#X_test = remove_feature('sentence_sentiment_score', X_test)

#X_train = remove_feature('in_title', X_train)
#X_test = remove_feature('in_title', X_test)


# X_train = remove_feature('token', X_train)
# X_test = remove_feature('token', X_test)
#X_train = remove_feature('token_pos_or_neg', X_train)
# X_train = remove_feature('token_sentiment_score', X_train)
#X_test = remove_feature('token_pos_or_neg', X_test)
# X_test = remove_feature('token_sentiment_score', X_test)

# del X
# del Y

# all_spans = train_spans + test_spans
# all_sent = X_train + X_test
# all_y = y_train + y_test
#one = []
#y_one = []
#one = X_test + one
#y_one = y_test + y_one
#two = X_test
#three =

crf = sklearn_crfsuite.CRF(
    algorithm='l2sgd',
    max_iterations=1000,
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
