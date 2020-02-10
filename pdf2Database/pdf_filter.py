import pickle
import spacy
import PyPDF2
import glob
import time
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import SpectralClustering



PDF_DIR = './temp/'
PDFS = glob.glob(PDF_DIR + '*.pdf')
assert len(PDFS) > 0, 'No pdf files found'


class pdf_transform:
    '''Class used to transform pdf documents, create doc2vec models based
    off of these pdfs, and label documents based off of clustering done on
    the document vectors

    This class is used to both create the initial doc2vec and clustering
    models and load these models to transform additional documents in the
    future.
    '''
    def __init__(self):
        '''
        Parameters:
        ----------
        pdf_text : list[str]
        list of tokenized/lemmatized strings from a document. This is the
        representation needed for doc2vec

        model : spacy model
        spacy model used to tokenize and lemmatize documents

        stopwords : set{str}
        stopwords removed from the document. In the future this may be
        augmented to include more words based on a tf-idf threshold

        books : list[str]
        List of file locations for PDFS longer than 50 pages
        They can pose a problem for doc2vec. These files
        are considered books and put in a different directory

        empties : list[str]
        File locations of empty files

        error_files : list[(str,Exception)]
        File locations of files that created some kind of
        problem during transformation. They are logged to be
        dealt with later

        text_matrix : list[list[str]]
        list of all of the pdf_text lists that are collected when processing
        multiple pdfs

        d2v_model : Doc2Vec
        Initialized by build_doc2vec_model() or load_d2vec_model()
        Used to transform docs into 50-D Vectors
        '''

        # Models
        self.pdf_text = []
        self.model = spacy.load("en_core_web_sm")
        self.d2v_model = None
        self.sc_model = None

        # Lists
        self.books = []
        self.empties = []
        self.error_files = []
        self.text_matrix = []
        self.vector_matrix = []

        # Stopwords
        self.stopwords = self.model.Defaults.stop_words
        self.stopwords = self.stopwords.union(set(string.punctuation))
        self.stopwords = self.stopwords.union(set(string.whitespace))

    def pickle_matrix(self):
        '''
        Pickles the document matrix. Useful in case you want to
        test how different doc2vec parameters affect document groups
        or if you had new functions to the class and want to test
        them out.

        Creating the document matrix is by far the longest process in
        transformation. So this cuts that time down.
        '''
        t = time.time()
        f_out = 'pdf_matrix_' + str(t) + '.pickle'
        pickle.dump(self.text_matrix, open(f_out, 'wb'))

    def load_matrix(self, f_in):
        '''Loads previously pickled text matrix'''
        self.text_matrix = pickle.load(open(f_in, 'rb'))

    def clear(self):
        '''Resets pdf_text to an empty list'''
        self.pdf_text = []

    def load_doc(self, file_in):
        '''
        - Loads single document into the class obj
        - Checks if doc is too large
        - Tokenizes and lemmatizes document
        - Initializes self.pdf_text as a list of the tokens in the document

        Parameters
        ----------
        file_in: str
        The file location of a pdf document

        If I end up wanting to add additional
        preprocessing then I will separate all of
        this and create a spacy pipeline
        '''
        try:
            pdf_file = open(file_in, 'rb')
            pdf = PyPDF2.PdfFileReader(pdf_file)
            pages = pdf.pages

            #  if len(pages)> 20: # Num of books: 119
            if len(pages) > 50:  # Num of books: 55
                print(file_in + ' is too large')
                self.books.append(file_in)

            elif len(pages) is 0:
                print(file_in + ' is empty')
                self.empties.append(file_in)

            else:
                for page in pages:
                    page_text = page.extractText()
                    doc = self.model(page_text)
                    for token in doc:
                        if token not in self.stopwords:
                            self.pdf_text.append(token.lemma_)

        except Exception as ex:
            self.error_files.append((file_in, str(ex)))

    def add_to_matrix(self):
        '''
        - Adds self.pdf_text list to self.text_matrix
        - clears self.pdf_text.

        This is used in transform_all_docs after each document is transformed
        '''
        self.text_matrix.append(self.pdf_text)
        self.clear()

    def load_all_docs(self, pdf_files):
        '''
        - Loads each file in the list of files passed to it
        - Preps the documents to be transformed

        Parameters
        ----------
        pdf_files: list[str]
        List of file locations of all of the pdfs being processed
        '''
        for f_in in pdf_files:
            self.load_doc(f_in)
            self.add_to_matrix()

    def build_doc2vec_model(self):
        '''
        - Builds doc2vec model based on documents in self.text_matrix
        - Saves model as a timestamped pickle file

        This function should only be called when you are first creating
        the doc2vec model.

        If you already have built the database and are just trying to
        label new documents then use load_model()
        '''

        docs = [TaggedDocument(doc, [i]) for i, doc
                in enumerate(self.text_matrix)]
        model = Doc2Vec(docs, vector_size=50, window=2, min_count=1, workers=3)

        t = time.time()
        f_out = 'd2vM_' + str(t) + '.bin'
        model.save(f_out)
        self.d2v_model = model

    def load_d2vec_model(self, f_in):
        '''
        Load doc2vec model

        Parameters
        ----------
        f_in : str
        file location of binary gensim doc2vec model
        '''

        self.d2v_model = Doc2Vec.load(f_in)

    def vectorize_documents(self):
        '''Transforms document into 50-D vector'''
        self.vec_matrix = [self.d2v_model.infer_vector(doc) for doc in
                           self.text_matrix]

    def build_sc_model(self):
        '''Builds an initial Spectral Clustering model'''
        sc = SpectralClustering(n_clusters=10, random_state=42)
        sc.fit(self.vec_matrix)
        self.sc_model = sc

        t = time.time()
        f_out = 'scM_' + str(t) + '.pickle'
        pickle.dump(sc, open(f_out, 'wb'))

    def load_sc(self, f_in):
        '''Load a premade Spectral Clustering model'''
        self.sc_model = pickle.load(open(f_in, 'rb'))

    def transform(
transformer = pdf_transform()
#transformer.load_all_docs(PDFS)
#transformer.build_doc2vec_model()

transformer.load_matrix('matrix')
transformer.load_d2vec_model('d2vM_1581277695.6772249.bin')
transformer.vectorize_documents()
transformer.build_sc_model()
