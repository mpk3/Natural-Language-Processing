import pickle
import spacy
import PyPDF2
import glob
import time
import string


class extract:
    """Extraction class:
    - Loads PDFs from directory
    - Separates books from articles
    - Tokenizes
    - Lemmatizes
    - Converts documents into lists of lemmas
    - Creates one large matrix of all of the documents
    - This matrix is then passed to the transformer
    """

    def __init__(self, files_in):
        """
        Parameters
        ----------

        FILE_DIR : str
        string representing directory of pdf files to be extracted
        ex: '/pdf/directory/'

        pdf_text : list[str]
        Temporary storage for tokenized/lemmatized documents

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

        titles : list[str]
        List of pdf titles that are going to be transformed.
        This is used to label pdfs after they have been transformed
        to retrieve their corresponding titles by its index
        """
        # Directories
        self.PDFS = glob.glob(files_in + '*.pdf')
        assert len(self.PDFS) > 0, 'No pdf files found'

        # Spacy Model
        self.model = spacy.load("en_core_web_sm")

        # Lists
        self.pdf_text = []
        self.books = []
        self.empties = []
        self.error_files = []
        self.titles = []

    def clear(self):
        '''Resets pdf_text to an empty list'''
        self.pdf_text = []

    def load_doc(self, file_in):
        """
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
        """
        try:
            pdf_file = open(file_in, 'rb')
            pdf = PyPDF2.PdfFileReader(pdf_file)
            pages = pdf.pages

            # Book page limit
            if len(pages) > 50:
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
            print('Error loading: ' + file_in)

    def add_to_matrix(self, f_in):
        '''
        - Adds self.pdf_text list to self.text_matrix
        - clears self.pdf_text.

        This is used in transform_all_docs after each document is transformed
        '''
        if len(self.pdf_text) > 0:
            self.text_matrix.append(self.pdf_text)
            self.titles.append(f_in)
            self.clear()

    def extract(self):
        '''
        - Loads each file
        - Text preprocessing
        - Builds final matrix

        Parameters
        ----------
        pdf_files: list[str]
        List of file locations of all of the pdfs being processed
        '''
        for f_in in self.PDFS:
            self.load_doc(f_in)  # Tokenization, Lemmatization etc.
            self.add_to_matrix(f_in)  # Build Matrix

extract = extract('./temp/')
extract.extract()
