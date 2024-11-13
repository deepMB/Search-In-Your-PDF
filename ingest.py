import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# Ensure temporary directory exists
if not os.path.exists("tmp"):
    os.makedirs("tmp")

def ingest_doc(file = None):
    documents = []
    if file :
        print("--------- Inside ingest if ------")
        tmp_location = os.path.join('tmp', file.name)
        print("--------- tmp_location ------")
        print(tmp_location)
        with open(tmp_location, "wb") as f:
            f.write(file.getbuffer())
        print("Pdf written in storage")
        loader = PyPDFLoader(tmp_location)
    else:
        for root,dirs,files in os.walk("docs"):
            for file in files:
                if file.endswith(".pdf"):
                    print(f"PDF file found --> {file}")
                    loader = PyPDFLoader(os.path.join(root,file))
    documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        texts, embeddings, persist_directory="db")
    
if __name__ == "__main__":
    ingest_doc()