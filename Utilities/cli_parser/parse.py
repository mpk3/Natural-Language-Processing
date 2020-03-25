from xpinyin import Pinyin
import jieba
import spacy
import argparse
import pickle
import re
# import gensim.downloader as api


# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cli",
                    help='Command line string parser;\
                    \'quit\' to exit; \'display\' for dependency graph',
                    action="store_true")
parser.add_argument("-d", "--display",
                    help='Display dependency graph; \'C-c C-c\' to exit',
                    action="store_true")
parser.add_argument("-s", "--similarity",
                    help='Gensim similarity',
                    action="store_true")
args = parser.parse_args()
parser = argparse.ArgumentParser()

# Spacy Initialization
nlp = spacy.load("en_core_web_sm")
doc = nlp('No sentence was loaded')

# Gensim Initialization
word_vectors = pickle.load(open('wv.pickle', 'rb'))

# word_vectors = api.load("glove-wiki-gigaword-100")
# pickle.dump(word_vectors, open('wv.pickle', 'wb'))


def mandarin_parse(sentence):
    ce_dict = pickle.load(open('ce_dict.pickle', 'rb'))
    p = Pinyin()
    words = jieba.cut(sentence)
    for word in words:
        print(word,
              p.get_pinyin(word, splitter=' ',
                           tone_marks='numbers'))
    return


def similarity(word):
    """
    Gensim similarity using built in function for quick
    similarity lookup. Can take a list of words or a 
    single word.

    Parameters:
    -----------

    word: []/string
    """
    # try:
    #    if type(eval(word)) == list:
    #        result = word_vectors.most_similar(positive=word)
    #except NameError as ne:
        
    #else:
    # result = word_vectors.most_similar(positive=[word])

    return # result


if args.cli:
    while True:
        sentence = input()
        if re.search("[\u4e00-\u9FFF]", sentence):
            mandarin_parse(sentence)
        elif sentence == 'quit' or 'q':
            break
        elif sentence == 'display':
            spacy.displacy.serve(doc, style="dep", page=True)
        else:
            doc = nlp(sentence)
            for token in doc:
                print(
                    "Token:{0:10s} Lemma:{1:10s} \
                    POS:{2:6s} Tag:{3:6s} \
                    Dep:{4:6s} Shape:{5:6s} \
                    Stop:{6:6s}"
                    .format(
                        token.text, token.lemma_,
                        token.pos_, token.tag_,
                        token.dep_, token.shape_,
                        token.is_stop))
elif args.display:
    sentence = input()
    if re.search("[\u4e00-\u9FFF]", sentence):
            mandarin_parse(sentence)
    elif sentence == 'quit':
        exit()
    else:
        doc = nlp(sentence)
        spacy.displacy.serve(doc, style="dep", page=True)

elif args.similarity:
    while True:
        sentence = input()
        if sentence == 'quit':
            break
        else:
            result = word_vectors.most_similar(positive=[sentence])
            print(result)
# if sentence == 'quit' or 'q':
        #     break
        # else:
        #     print(similarity(sentence))

else:
    print('Requires option flag\nPython parse.py --help for options ')
