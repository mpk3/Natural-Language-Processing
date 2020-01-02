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


# f = '/home/mpk3/School/MS_Final_Project/project/results/2tpe300.txt'
# f = '/home/mpk3/School/MS_Final_Project/project/results/trial_1/xab'

def generate_baseline(y_test):
    baseline = []
    for tag_list in y_test:
        baseline.append(['o' for i in range(len(tag_list))])
    return baseline

def flatten_sent_embedding(sent):
    for token_dict in sent:
        emb = token_dict.pop('embedding')
        emb_list = list(enumerate(emb))
        for i, flt in emb_list:
            token_dict[str(i)] = float(flt)  # Also converts to float
    return sent


fin = '/home/mpk3/School/MS_Final_Project/project/results/trial_2/'

files = glob.glob(fin + "*.pickle")
half = files[0:184]
X = []
Y = []

# files_test = files[0:3]
for article in half:  # files_test:
    sentences = pickle.load(open(article, "rb"))
    for X_i in sentences:
        Y_i = [y.pop('label') for y in X_i]
        X_i = flatten_sent_embedding(X_i)
        X.append(X_i)
        Y.append(Y_i)



X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.20,
                                                    random_state=42)
del X
del Y

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=50,
    all_possible_transitions=True)


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
baseline = generate_baseline(y_pred)
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
