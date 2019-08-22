import spacy
import Counter

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

    def __init__(self, paths=None, new=False):
        """Args:
        path (List[String]): List of paths for input
        new (Boolean): Determines whether or not new data is being
                       created or whether old data is being loaded
        model (Language model) : pretrained spaCy model
        nlp (Language object): language object with loaded model
        """
        self.paths = paths
        self.model = None
        self.nlp = None
        self.tokens = None
        self.counts = None

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

    def create_tokens(self):
        """Get raw token list of documents
        """
        all_tok = []
        # Input
        for filein in self.paths:
            tokens = []
            with open(filein, 'r') as text:
                text = text.read()
                doc = self.nlp(text)
                tokens = [str(token.text) for token in doc
                          if
                          token.is_space is not True
                          and
                          token.is_punct is not True
                          and
                          token.is_stop is not True]
            all_tok.append(tokens)
        self.tokens = all_tok


def create_counts(self):
    self.counts = Counter(self.tokens)


def most_common(self, n=5):
    return self.counts.most_common(n)


def main():
    gadget = Inspector(['test_data/test.txt'])
    gadget.english()
    gadget.create_tokens()
    gadget.create_counts()
    gadget.most_common(n=3)
    return gadget


if __name__ == "__main__":
    john = main()
