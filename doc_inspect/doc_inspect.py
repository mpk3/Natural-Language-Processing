import spacy
import pandas as pd
import datetime
import string
# import numpy as np
import matplotlib.pyplot as plt
# from spacy import displacy


def create_data(paths, create_file=False, stopwords=True):
    """Load a list of document paths;
    Returns an inverted index to graph from

    Keyword Arguments:
    paths -- list [] ; paths to corpus
    create_file -- Boolean; Creates timestamped csv for future use;\
                   default=False
    stop_words  -- Boolean; True removes stopwords as index is built;\
                   default=True
    """
    # Inverted Index
    inv_index = pd.DataFrame(columns=['document', 'token', 'POS'])
    # load model
    nlp = spacy.load("en_core_web_sm")
    # Create output file
    if (create_file):
        now = datetime.datetime.now()
        ymdhms = str(now.year)+'_' + \
            str(now.month)+'_' + \
            str(now.day)+'_' + \
            str(now.hour)+'_' + \
            str(now.minute)+'_' + \
            str(now.second)+'_'
        filename = ymdhms+str(len(paths))+'.csv'
        fileout = open(filename, 'w')
    # Begin reading corpora
    n = 0
    for path in paths:
        # Input
        with open(path, 'r') as text:
            text = text.read()
            # NLP
            doc = nlp(text)
            # Build Index
            for token in doc:
                if (stopwords):
                    if (token.is_stop) or\
                       (token.text in set(string.punctuation)) or \
                       (token.text in set(string.whitespace)):
                        next
                    else:
                        inv_index.loc[n] = [path,
                                            token.text.lower(),
                                            token.pos_]
                        n = n + 1
                        if (create_file):
                            fileout.writelines(path + ',' +
                                               token.text.lower() + ',' +
                                               token.pos_)
                else:
                    inv_index.loc[n] = [path, token.text.lower(), token.pos_]
                    n = n + 1
                    if (create_file):
                        fileout.writelines(path + ',' +
                                           token.text.lower() + ',' +
                                           token.pos_)
    if (create_file):
        fileout.close()
    return inv_index


def inv_index_token(df):
    '''
    Input from create_data()
    Returns  inv index for tokens '''
    df_II_c = df.groupby('document')\
                .token.value_counts()\
                      .to_frame()\
                      .rename(columns={'token': 'count'})

    return df_II_c


def inv_index_POS(df):
    '''
    Input from create_data()
    Returns  inv index for  POS'''
    df_II_c = df.groupby('document')\
                .POS.value_counts()\
                    .to_frame()\
                    .rename(columns={'POS': 'count'})

    return df_II_c


# def remove_stopwords(df):
    # not finished
    # nlp = spacy.load("en_core_web_sm")
    #    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    # nlp.stop_words.STOP_WORDS
    #   df_no_stops = df[~df[
    # stopindex = [nlp(df['token'][i])[0].is_stop\
    #            for i in range(len(df['token']))]


def plot_nlargest(series, n=5, index_level=0):
    """Returns a bar plot of top n tokens/pos

    Keyword Arguments:
    series -- pd Series object to be plotted
    n -- top n tokens/POS plotted for each document
    index_level -- default should be 0; can be manipulated for different\
    results
    """
    outplot = series.groupby(level=index_level)\
                    .nlargest(n)\
                    .reset_index(level=index_level, drop=True)
    outplot.plot.bar(title='Top ' + str(n))
    return outplot.to_frame()


def standard_make(filein):
    """Utility function for the normal procedures I go through while inspecting \
    text

    Keyword Arguments:
    filein -- List of input paths
    """
    if type(filein) is list:
        df_no_stops = create_data(filein)
        df_tok_c = inv_index_token(df_no_stops)
        df_pos_c = inv_index_POS(df_no_stops)
        return df_no_stops, df_tok_c, df_pos_c
    else:
        print('Filein must be List [String]')
        return None


# Quick Driver
# run ner_displacy.py
# filein = ['test_data/CofW.txt']
# df, df_tok, df_pos = standard_make(filein)
# tokplot = plot_nlargest(df_tok_c['count'], 5, 0)
# plt.show()







# run ner_displacy.py
# filein = ['test_data/CofW.txt', 'test_data/faust.txt']
# df = create_data(filein)
# df_tok_c = inv_index_token(df)
# df_pos_c = inv_index_POS(df)

# tokplot = plot_nlargest(df_tok_c['count'], 5, 0)
# posplot = plot_nlargest(df_pos_c['count'], 5, 0)

# plt.show()
