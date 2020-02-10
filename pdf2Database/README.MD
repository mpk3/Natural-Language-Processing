# pdf2Database

Problem: I download a ton of papers off of the internet and they often have obscure names that are only reference numbers. These files typically pile up in my ~/Downloads folder and get lost.

Solution: Create a local ETL process to deal with the pdfs. Use a combination of OCR(if necessary), PyPDF2, doc2vec, dimensionality reduction,  and SQL to build a database and catalog these files.
