import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from loader import carregar_documentos_json
from embedding_store import criar_chroma, carregar_chroma

load_dotenv()

def criar_chatbot():
    persist_dir = "chroma_db"
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("🔄 Criando base vetorial...")
        documentos = carregar_documentos_json("dados/conteudo_paginas_tratado.json")
        print(documentos)
        vectorstore = criar_chroma(documentos, persist_dir)
    else:
        print("✅ Carregando base vetorial existente...")
        vectorstore = carregar_chroma(persist_dir)


    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    return chain

if __name__ == "__main__":
    chatbot = criar_chatbot()
    
    while True:
        pergunta = input("Faça uma pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        resposta = chatbot(pergunta)
        print("\n🤖 Bot:", resposta["result"])
        fontes = [doc.metadata['titulo'] for doc in resposta["source_documents"]]
        print("📚 Fontes:", fontes)
        print("-" * 50)
