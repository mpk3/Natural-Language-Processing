import scipy.stats
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


# Deprecated; Will be used later
def flatten_sent_embedding(sent):
    for token_dict in sent:
        emb = token_dict.pop('embedding')
        emb_list = list(enumerate(emb))
        for i, flt in emb_list:
            token_dict[str(i)] = float(flt)  # Also converts to float
    return sent


TRIAL = '/trial1/'
fin = LAB_DAT + TRIAL + "*.pickle"

files = glob.glob(fin)
amount = files  # files[0:184]
X = []
Y = []

# files_test = files[0:3]
for article in amount:  # files_test:
    sentences = pickle.load(open(article, "rb"))
    for X_i in sentences:
        Y_i = [y.pop('class') for y in X_i]
        #X_i = flatten_sent_embedding(X_i)
        X.append(X_i)
        Y.append(Y_i)



X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.20,
                                                    random_state=42)
train_spans, X_train = separate_spans(X_train)
test_spans, X_test = separate_spans(X_test)

del X
del Y
all_spans = train_spans + test_spans
all_sent = X_train + X_test

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
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
                        cv=3,
                        verbose=1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)

y_pred = rs.predict(X_test)

metrics.flat_f1_score(y_test, y_pred, average='weighted')
metrics.flat_f1_score(y_test, baseline, average='weighted')
metrics.flat_accuracy_score(y_test, y_pred)
metrics.flat_recall_score(y_test, y_pred)


#
#
#
#  Optimization:


crf = rs.best_estimator_
y_pred = crf.predict(X_test)
