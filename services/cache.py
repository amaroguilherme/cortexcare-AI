import uuid

from config import load_index, load_vector_db
from services.embedding_model import TransformerHandler


class CacheHandler:
    def __init__(self):
        self.db = load_vector_db()
        self.index = load_index()

    def get_from_cache(self, input: list):
        results = self.index.query(
            vector=input,
            top_k=1,
            include_metadata=True
        )
        
        return results
    
    def write_on_cache(self, input: str, output: str):
        embeddings = TransformerHandler().encode_input(input)
        
        self.index.upsert([
            {
                "id": str(uuid.uuid4()),
                "values": embeddings,
                "metadata": {
                    "question": input,
                    "answer": output
                }
            }
        ]) #TODO: IMPROVE ERROR HANDLING
