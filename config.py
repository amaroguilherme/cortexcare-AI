import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

def load_vector_db():
    return Pinecone(
        api_key=os.getenv("VEC_DB_API_KEY")
    )
    
def load_index():
    pc = load_vector_db()
    index_name = os.getenv("VEC_DB_INDEX_NAME")
    
    return pc.Index(name=index_name)
    
def load_trained_model():
    return SentenceTransformer('all-mpnet-base-v2')

def get_language_model_api_key():
    return os.getenv("LANGUAGE_MODEL_API_KEY")

def get_language_model_url():
    return os.getenv("LANGUAGE_MODEL_URL")
