## Extract Snippets from Documents

Scripts used in order to extract snippets from the retrieved documents. In order for the script to be executed with sucess, we need the top 100 documents and the output of the Document Retrieval model.

#### Specification

- **Datasets**: Produced datasets (text files).
- **PubMeds**: Data related to PubMed documents.
- **main.py**: Script to run.
- **dataset.py**: Script that creates datasets based on BioASQ format.
- **improve_test_set**: Functions used in order to create the final test set.
- **download_nltk.py**: Run this script in order to download the nltk library.