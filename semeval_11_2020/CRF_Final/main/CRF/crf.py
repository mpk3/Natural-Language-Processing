# import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pickle
import glob
from sklearn.model_selection import train_test_split


LAB_DAT = '../labeled_data'


def generate_baseline(y_test):
    '''Creates a baseline by labeling all results 'o' for other
    Gives a metric to work against when messing around with the data
    '''
    baseline = []
    for tag_list in y_test:
        baseline.append(['o' for i in range(len(tag_list))])
    return baseline


def separate_spans(X):
    '''Token location information is in all the data. 
    It is needed to map it back to spans of text in order to send
    results to the competition. This function removes those spans
    so they are not interpreted as features my the CRF
    '''
    X_span = []
    for sentence in X:
        spans = [feat_map.pop('span') for feat_map in sentence]
        X_span.append(spans)
    return X_span, X


def separate_articles(X):
    '''Same as separate_spans() but for the article reference numbers'''
    X_articles = []
    for sentence in X:
        articles = [feat_map.pop('article') for feat_map in sentence]
        X_articles.append(articles)
    return X_articles, X


def one_hot_encoder(feature_key, X):
    '''DEPRECATED
    Creates one hot encoding for whatever key is passed to it.'''
    j = set()
    x = [[j.add(token[feature_key]) for token in sentence]
         for sentence in X]
    del x
    one_hot = OneHotEncoder()
    j = np.array(list(j)).reshape(-1, 1)
    one_hot.fit(j)
    return one_hot


def encode(encoder_list, X):
    '''DEPRECATED
    Changes X so it is one_hot encoded'''

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
    '''DEPRECATED
    CRF suite has very particular requirements for the shape of
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
    '''This function is currently defunct. I wont need it operational
    until I  begin sending results to the competition
    '''
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
    '''Removes all features not in feature_set from X. This
    is used during testing to easily remove features and compare
    results'''
    keys = set(X[0][0].keys())
    for key in keys:
        if key not in feature_set:
            X = remove_feature(key, X)
    return X


# Driver Code
# Test Sets
# N-Grams 

unigram =\
    ['pos', 'token',  'main_is_stop', 'main_is_lower', 'vader_token']
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

# CURRENT OUTPUT MODEL TEST PHASE 3
v0 = fourgram + ['vader_token']
v1 = fourgram + ['vader_prev', 'vader_next']
v2 = fourgram + ['vader_prev_prev', 'vader_next_next']

n_gram_trials = [v0, v1, v2]


trials = n_gram_trials
TRIAL = '/trial10/'
fin = LAB_DAT + TRIAL + "*.pickle"
files = glob.glob(fin)


i = 0
for trial_feature_list in trials:
    X = []
    Y = []

    for article in files:  # files_test:
        sentences = pickle.load(open(article, "rb"))
        sentences = sentences[1:]  # Removes title sentence
        for X_i in sentences:
            Y_i = [y.pop('class') for y in X_i]
            X.append(X_i)
            Y.append(Y_i)

    test_feat_set = set(trial_feature_list)
    Z = test_function(test_feat_set, X)
    X_train, X_test, y_train, y_test = train_test_split(Z,
                                                        Y,
                                                        test_size=0.20,
                                                        random_state=42)
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        max_iterations=1000,
        all_possible_states=True,
        all_possible_transitions=True,
        verbose=True)

    # set_f_out = str(i) + '_features.txt'
    # sys.stdout = open(set_f_out, "w")

    # print (Z[0][0].keys())
    # sys.stdout.close()
    iterations = str(i) + '_iterations.txt'
    # sys.stdout = open(iterations, "w")
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    # sys.stdout.close()

    # results = str(i) + '_results.txt'
    # sys.stdout = open(results, "w")
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)
    print(metrics.flat_classification_report(y_test, y_pred))
    # sys.stdout.close()
    i = i + 1
