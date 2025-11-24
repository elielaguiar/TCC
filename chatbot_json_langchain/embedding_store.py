from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from openrouter_embeddings import OpenRouterEmbeddings


def criar_chroma(documentos, persist_dir="chroma_db"):
    embeddings = OpenRouterEmbeddings.from_env()
    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print("Persistindo base vetorial em disco...")
    vectorstore.persist()
    return vectorstore


def carregar_chroma(persist_dir="chroma_db"):
    embeddings = OpenRouterEmbeddings.from_env()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
