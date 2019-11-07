from joblib import load

# Sentiment Analysis

# Language Identification
l_clf = load('/home/mpk3/Natural_Language_Processing/' +
             'NB/results/lang_clf.joblib')
# Producing wrong results
# Builtin
stdin = input("Enter Text:")
print(stdin)
stdin = [stdin]
# This may be where the problem is 
languages = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ja', 'nl', 'pl', 'pt', 'ru']

lang_out = languages[l_clf.predict(stdin)[0]]

if lang_out is 'en':
    sentiment = ['neg', 'pos']
    s_clf = load('/home/mpk3/Natural_Language_Processing/' +
                 'NB/results/sent_clf.joblib')
    print('en : ' + sentiment[s_clf.predict([stdin])])
else:
    print(lang_out)
