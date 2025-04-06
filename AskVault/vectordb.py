import os
from pinecone import Pinecone, ServerlessSpec  # ✅ Pinecone SDK
from langchain.vectorstores import Pinecone as PineconeLangChain  # ✅ LangChain Pinecone
from decouple import config
from .helper import download_hugging_face_embeddings

# Load API Key and Environment from .env
embeddings = download_hugging_face_embeddings()
PINECONE_API_KEY = config('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
PINECONE_ENV = config('PINECONE_ENV', default="gcp-starter")  # Adjust environment if needed

# Initialize Pinecone client (SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)  # ✅ Corrected initialization

# Define index name
index_name = "askvault"

def clear_pinecone():
    """Delete all existing vectors from the Pinecone index, if it exists."""
    index = pc.Index(index_name)
    stat = index.describe_index_stats()
    if index and stat.get("total_vector_count", 0) !=0:
        index.delete(delete_all=True, namespace="")
        print(f"Cleared all vectors from index: {index_name}")
    else:
        print(f"Index '{index_name}' does not exist. Skipping deletion.")

def get_vectordb_instance():
    """Initialize or retrieve a Pinecone vector database instance"""
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name, 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ), 
            dimension=384, 
            metric="cosine"
        ) 

    vectorstore = PineconeLangChain.from_existing_index(index_name, embeddings)  # ✅ Use LangChain's Pinecone

    print(f"VectorStore initialized for index: {index_name}")
    return vectorstore
