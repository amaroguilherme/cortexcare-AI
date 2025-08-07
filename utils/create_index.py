import logging
import os

from dotenv import load_dotenv
from pinecone import ServerlessSpec

from config import load_vector_db

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

load_dotenv()

pc = load_vector_db()

index_name = os.getenv("VEC_DB_INDEX_NAME")

log.info('Creating Index (if it doesnt exist)...')
if index_name not in pc.list_indexes().names():
    try:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        log.info(f'Created Index: {index_name}')
    except Exception as e:
        log.info(f'Failure while creating Index: {e}')