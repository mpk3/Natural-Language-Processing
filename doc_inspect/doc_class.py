import spacy
import subprocess

class Inspector():
    """This is a class for holding the basic data created when
    when reading and parsing a text. It holds all of the the basic
    attributes that the other classes work off of:

    Attributes:
        paths (List[String]): List of paths for input
        new (Boolean): Determines whether or not new data is being
                       created or whether old data is being loaded
        model (Language model) : pretrained spaCy model
        nlp (Language object): language object with loaded model

    """

    def __init__(self, path=None, new=False):
        """Args:
        paths (List[String]): List of paths for input
        new (Boolean): Determines whether or not new data is being
                       created or whether old data is being loaded
        model (Language model) : pretrained spaCy model
        nlp (Language object): language object with loaded model
        """
        self.paths = path
        self.model = None
        self.nlp = None

    def summary(self):
        """Returns corpora name"""
        print("Corpora: " + str(self.paths) + "\n "
              "Model: " + str(self.model) + "\n "
              "Version: " + str(spacy.__version__))

    def english(self):
        """ Creates model instance from an English multi-task CNN
        trained on OntoNotes"""
        self.model = 'en_corge_web_sm'
        self.nlp = spacy.load('en_core_web_sm')

    def german(self):
        """  German multi-task CNN trained on the TIGER and
        WikiNER corpus."""
        self.model = 'de_core_news_sm'
        self.nlp = spacy.load('de_core_news_sm')

    def quickParse(self):
        """ Attempt at using some quick unix commands to find
        token counts; plan to check the results against nlp method
        """
        if(self.model == 'de_core_news_sm'):
            stopwords = spacy.lang.de.stop_words.STOP_WORDS
        else:
            stopwords = spacy.lang.en.stop_words.STOP_WORDS
        # print(stopwords)
        # Pipe
        #tokenize = subprocess.Popen(['egrep', self.paths])
        tok = subprocess.Popen(['egrep', '-o', '\S+', self.paths[0]],
                               stdout=subprocess.PIPE)
        print(tok.stdout)
        sort = subprocess.Popen(['sort'],
                                stdin=tok.stdout,
                                shell=True,
                                stdout=subprocess.PIPE)
        print(sort.stdout)
        unique = subprocess.Popen(['uniq', '-c'],
                                  stdin=sort.stdout,
                                  shell=True,
                                  stdout=subprocess.PIPE)
        print(unique.stdout)
        unigram_counts = subprocess.Popen(['sort', '-nr'],
                                          stdin=unique.stdout,
                                          shell=True, stdout=subprocess.PIPE)
        print(format(unigram_counts.stdout))
        end_of_pipe = unigram_counts.stdout

        print('Included files:')
        for line in end_of_pipe:
            print(line.decode('utf-8').strip())
