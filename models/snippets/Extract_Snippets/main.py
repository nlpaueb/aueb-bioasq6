import json, os, dataset, errno
from Bio.Entrez import efetch
import pickle
import improve_test_set as improve

error_docs = 0

# Collect the ids of relevant documents and retrieve their xml form in order to extract the abstract of each document
def scrap_PubMeds(documents, dir_docs):

    ids = []    # Here we save the ids of the desired documents

    # retrieve ids from relevant documents
    for document_list in documents:
        for document in document_list:
            ids.append(document.rsplit('/', 1)[-1])

    print("Number of total PubMeds to be extracted: ", len(ids))
    doc_counter = 0
    for docid in ids:
        doc_counter += 1
        print(docid, " : ", doc_counter, "/", len(ids))

        # Create folder for the specific document
        try:
            os.makedirs("PubMeds/Abstracts/" + docid)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        f = open("PubMeds/PubMeds_ids", 'a')
        f.write(docid+ "\n")
        f.close()

        get_doc_info(docid, dir_docs)


def get_doc_info(docid, dir):

    try:
        doc = dir[str(docid)]
        title = doc['title']
        abstract = doc['abstractText']
    except:
        global error_docs
        error_docs += 1
        title = ""
        abstract = ""
        print("Error in doc: ", docid)


    if abstract != "":
        # Save abstract
        f = open("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'w')
        f.write(abstract)
        f.close()

    if title != "":
        # Save title
        f = open("PubMeds/Abstracts/" + docid + "/" + docid + "_title.txt", 'w')
        f.write(title)
        f.close()





# Retrieve XML form for a specific document
# We just need to insert the id of the desired document
def get_XML(doc_id):
    handle = efetch(db='pubmed', id=doc_id, retmode='text', rettype='xml')
    return handle.read()


questions = []
documents = []
questions_ids = []

# Load pickle file containing info about the available docs of the specific test batch
print("Loading pickle file with document info...")
with open("top100.pkl", "rb") as input_file:
    dir_docs = pickle.load(input_file)
print("File loaded with success!")

print("Loading json file with questions and candidate relevant documents...")


file_name = "DocRetrievalModel.json"

with open(file_name)as data_file:
    data = json.load(data_file)
print("File loaded with success!")


print("Retrieving relevant documents for each question...")
for question in data['questions']:
    questions.append(question.get('body'))
    documents.append(question.get('documents'))
print("#questions: ", len(questions))

scrap_PubMeds(documents, dir_docs)


for question in data["questions"]:
    snippets = []
    questions_ids.append(question.get('id'))
    snippets.append(question.get('snippets'))
    temp_docs = question.get('documents')

    for document in temp_docs:
        docid = document.rsplit('/', 1)[-1]
        if os.path.isfile("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt"):
            dataset.createTestSetForm(question.get('id'), question.get('body'), docid)

print("Errors: ", error_docs)

print("Creating final form for test set...")
improve.improve_dataset("Datasets/test.txt")