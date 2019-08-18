import spacy
import pandas as pd
import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# from spacy import displacy


def create_data(paths, create_file):
    '''
    Load a list of document paths;
    Returns an inverted index to graph from
    '''
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
                inv_index.loc[n] = [path, token.text.lower(), token.pos_]
                if (create_file):
                    fileout.writelines(path + ',' +
                                       token.text.lower() + ',' +
                                       token.pos_)
                    n = n + 1
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


def remove_stopwords(df):
    '''not finished'''
    nlp = spacy.load("en_core_web_sm")
   # stopindex = [nlp(df['token'][i])[0].is_stop for i in range(len(df['token']))]



def plot_nlargest(series, n, index_level):
    '''
    series : pd Series object to be plotted
    n: top n tokens/POS plotted for each document
    index_level: default should be 0; can be manipulated for different results
    Returns a bar plot of top n tokens/pos
    '''
    outplot = series.groupby(level=index_level)\
                    .nlargest(n)\
                    .reset_index(level=index_level, drop=True)
    outplot.plot.bar()
    return outplot.to_frame()


# Quick Driver

# run ner_displacy.py
# filein = ['test_data/CofW.txt', 'test_data/faust.txt']
# df = create_data(filein)
# df_tok_c = inv_index_token(df)
# df_pos_c = inv_index_POS(df)

# tokplot = plot_nlargest(df_tok_c['count'], 5, 0)
# posplot = plot_nlargest(df_pos_c['count'], 5, 0)

# plt.show()
