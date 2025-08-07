import logging
import json
import uuid

from config import load_index, load_trained_model, load_vector_db

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

pc = load_vector_db()

index = load_index()

log.info('Loading pre-trained model...')
pre_trained_model = load_trained_model()
log.info('Finished loading pre-trained model')

with open("batches.json", "r", encoding="utf-8") as file:
    data = json.load(file)
data_to_ingest = data['MANUAL_QUESTION_ANSWER']

for d in data_to_ingest:
    log.info('Embedding text...')
    embeddings = pre_trained_model.encode(d['question'], convert_to_tensor=True)
    log.info('Finished embedding text')
    
    log.info('Starting upsert...')
    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": embeddings,
            "metadata": {
                "question": d['question'],
                "answer": d['answer']
            }
        }
    ])
    log.info('Finished upsert')