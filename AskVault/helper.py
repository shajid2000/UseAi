from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from .prompt import prompt_template
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle


def download_hugging_face_embeddings(local_path="LLM-Models/embeddings/embeddings.pkl"):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if os.path.exists(local_path):
        print("Loading embeddings from local storage...")
        with open(local_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Downloading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        with open(local_path, "wb") as f:
            pickle.dump(embeddings, f)
    
    return embeddings


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

def handle_uploaded_file(uploaded_file):
    """ Save uploaded file and return its path """
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    default_storage.save(file_path, ContentFile(uploaded_file.read()))
    return file_path

def extract_text(file_path, file_name):
    """ Extract text from PDF or text file """
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_name.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        return None 
        
    return loader.load()


def text_split(extracted_data):
    """ Split extracted text into smaller chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)