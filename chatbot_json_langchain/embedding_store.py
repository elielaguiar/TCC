from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def criar_chroma(documentos, persist_dir="chroma_db"):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore

def carregar_chroma(persist_dir="chroma_db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
