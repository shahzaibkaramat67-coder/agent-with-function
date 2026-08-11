from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv

load_dotenv()


def createChromaDB(file):
    print("file", file)
    if not file:
        raise ValueError("No document content was provided for indexing.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(file)

    model_name = os.getenv(
        "EMBADDING_MODEL_SENTENCE", "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("model_name", model_name)
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # persist_directory = os.path.join("data", "chroma_db")
    # os.makedirs(persist_directory, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="Databse/croma_db",
    )
