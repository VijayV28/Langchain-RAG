from dotenv import load_dotenv
from langchain_community.utilities import ApifyWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import cohere

import time

# Loading the API keys
load_dotenv("../.env")

# Initializing the website crawler
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://cohere.com/llmu"}],
        "maxCrawlDepth": 2,
    },
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=(
            dataset_item["text"] if dataset_item["text"] else "No content available"
        ),
        metadata={
            "title": dataset_item["metadata"]["title"],
            "source": dataset_item["url"],
        },
    ),
)


docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, length_function=len
)
docs_split = text_splitter.split_documents(docs)
len(docs_split)

# Embedding
embedding = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain")  # 1024

# Deleting an existing dataset
# DeepLake.force_delete_by_path(f"hub://{org_id}/{db_name}")

# DeepLake Vector Store
org_id = "vijayv2807"
db_name = "cohere-llmu-data"
dataset_path = f"hub://{org_id}/{db_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embedding)

db.add_documents(docs_split)


# Pinecone Vector Store
index_name = "cohere-llmu-data"
pc = Pinecone()

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embedding)
vector_store.add_documents(docs_split)
