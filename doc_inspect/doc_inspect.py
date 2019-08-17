import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from spacy import displacy





def create_data(paths):
    '''
    Load a list of document paths;
    Returns an inverted index to graph from
    '''
    # Inverted Index
    inv_index = pd.DataFrame(columns=['document', 'token', 'POS'])
    # load model
    nlp = spacy.load("en_core_web_sm")
    n = 0
    # Begin reading corpora
    for path in paths:
        # Input
        text = open(path, 'r')
        text = text.read()
        # NLP
        doc = nlp(text)
        # Build Index
        for token in doc:
            inv_index.loc[n] = [path, token.text.lower(), token.pos_]
            n = n + 1
    return inv_index


def inv_index_token(DataFrame):
    ''' Returns  inv index for tokens '''
    df_II_c = df.groupby('document')\
         .token.value_counts()\
         .to_frame()\
         .rename(columns={'token':'count'})

    return df_II_c


def inv_index_POS(DataFrame):
    ''' Returns  inv index for  POS'''
    df_II_c = df.groupby('document')\
                .POS.value_counts()\
                    .to_frame()\
                    .rename(columns={'POS': 'count'})

    return df_II_c


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
