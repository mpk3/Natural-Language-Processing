import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# Inspecting data "bunch" ~ sklearn term
twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

# For speed classes are indices
twenty_train.target[:10]
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

# Tokenizing text with scikit-learn
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
# Supports counts of N-grams of words or consecutive characters
# Looking at dictionary of indices
count_vect.vocabulary_.get(u'algorithm')

# Term frequencies
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
# fit_transform is a better method; cuts out redundant processing
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Naive Bayes Classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Predict the outcome on a new document
# Transforming new text
docs_new = ['God is love', 'OpenGl on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Prediction
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

# Building a pipeline
# Parameter names are arbitrary; used for grid search on hyperparams
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),])

text_clf.fit(twenty_train.data, twenty_train.target)

# Evalutation of the performance on the test set
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

# Comparing against SVM; Better for classification but slower?
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge',
                          penalty='l2',
                          alpha=1e-3,
                          random_state=42,
                          max_iter=5,
                          tol=None))])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

# Utilities for detailed performance analysis
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))
metrics.confusion_matrix(twenty_test.target, predicted)

# Paramater Tuning using grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)}

# n_jobs parameter looks at CPU count and then runs models in parallel
# n_jobs = -1 will use all available CPUs
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

# Fitting on smaller subset for sake of speed
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
twenty_train.target_names[gs_clf.predict(['God is love'])[0]]

# best_score_ and best_params_ store best mean score and corresponding settings
gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# Detailed summary of results
gs_clf.cv_results_
