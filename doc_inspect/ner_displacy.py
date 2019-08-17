import spacy
import pandas as pd

from spacy import displacy


filein = ['test.txt', 'test2.txt']

def create_data(paths):
    '''
    Load a list of documents;
    Returns an inverted index to graph from
    '''
    
    # Inverted Index
    inv_index = pd.DataFrame()#columns=['document','token','POS'])
    # load model
    nlp = spacy.load("en_core_web_sm")
    

    for path in paths:
        
        # Input
        text = open(path,'r')
        text = text.read()
        
        # NL
        doc = nlp(text)
        
        # Build Index
        token_tuples = []
        
        tok = []
        pos = []
        n = 0
        
        for token in doc:
            #token_tuples.append[(path,token.text,token.pos))]
            #s = pd.concat([s1]
            #inv_index = inv_index.append(s1, keys=path)
            tok.append(token.text)
            pos.append(token.pos_)
            n = n + 1
        
    s1 = pd.DataFrame({'token':tok,'pos':pos} ,index=path)    
            
    return inv_index , s1


        
        
