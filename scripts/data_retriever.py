from pinecone import Pinecone, ServerlessSpec
import streamlit as st

from langchain.vectorstores import DeepLake
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank


@st.cache_resource()
def data_lake(dataset_type: str = "deeplake"):

    # Embedding
    embedding = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain")

    # Databases
    index_name = "cohere-llmu-data"
    if dataset_type == "deeplake":
        org_id = "vijayv2807"
        dataset_path = f"hub://{org_id}/{index_name}"
        db = DeepLake(dataset_path=dataset_path, embedding=embedding, read_only=True)
    else:
        pc = Pinecone()
        index_name = "cohere-llmu-data"
        index = pc.Index(index_name)
        db = PineconeVectorStore(index=index, embedding=embedding, read_only=True)

    # Retriever
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 25, "fetch_k": 8, "distance_metric": "cos"},
    )

    # Document Compressor
    compressor = CohereRerank(
        model="rerank-english-v3.0",
        top_n=5,
        user_agent="langchain",  # top_n -> No of documents to return
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return db, retriever, compression_retriever
