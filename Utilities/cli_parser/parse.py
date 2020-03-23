import spacy
import argparse

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cli",
                    help='Command line string parser; \'quit\' to exit; \'display\' for dependency graph',
                    action="store_true")
parser.add_argument("-d", "--display",
                    help='Display dependency graph; \'C-c C-c\' to exit',
                    action="store_true")
args = parser.parse_args()

# Spacy Initialization
nlp = spacy.load("en_core_web_sm")
doc = nlp('No sentence was loaded')

if args.cli:
    while True:
        sentence = input()
        if sentence == 'quit':
            break
        if sentence == 'display':
            spacy.displacy.serve(doc, style="dep", page=True)
        else:
            doc = nlp(sentence)
            for token in doc:
                print("Token:{0:10s} Lemma:{1:10s} POS:{2:6s} Tag:{3:6s} Dep:{4:6s} Shape:{5:6s}"
                      .format(token.text, token.lemma_, token.pos_, token.tag_,
                              token.dep_, token.shape_, token.is_stop))
elif args.display:
            if sentence == 'display':
                spacy.displacy.serve(doc, style="dep", page=True)
else:
    print('Requires option flag\nPython parse.py --help for options ')
